"""Real-data CUDA contract test for PC-MR-RPN plus PC-MOC-FD."""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import DEVICE, seed_all
from common.dataset import (
    build_copy_paste_pool,
    build_training_datasets,
    collate_fn,
    compute_reliability_threshold,
)
from common.metrics import get_metric_fn
from common.model import (
    attach_pc_micro_object_feature_teacher,
    attach_pc_micro_rescue_rpn_teacher,
    build_model,
)
from common.train_utils import _backward_with_disjoint_pcgrad


DEFAULT_TEACHER = (
    ROOT
    / ".runtime/kaggle/cbl_iterative_train_fair20/output/tod_output/runs"
    / "sa_alw_full__cbl__irtw0.5ir1s0.3__la_loss__seed42__cbl_iterative_train_fair20"
    / "best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "runs/pc_mr_moc_technical_smoke_seed42.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=20)
    return parser.parse_args()


def _build(reliability_threshold: float) -> torch.nn.Module:
    return build_model(
        metric_fn=get_metric_fn("sa_alw_full"),
        placement="la_loss",
        reliability_thr=reliability_threshold,
        box_loss_type="cbl",
        box_loss_warmup_epochs=0,
        cbl_refine_train_weight=0.5,
        cbl_refine_steps=1,
        cbl_refine_blend=1.0,
        cbl_refine_score_threshold=0.30,
        cbl_alpha=5.0,
        cbl_num_bins=6,
        cbl_grid_beta=1.0,
        cbl_um_weight=1.0,
        transform_min_sizes=(640,),
        transform_max_size=800,
    ).to(DEVICE)


def _move_batch(images, targets):
    return (
        [image.to(DEVICE) for image in images],
        [
            {
                key: value.to(DEVICE) if torch.is_tensor(value) else value
                for key, value in target.items()
            }
            for target in targets
        ],
    )


def _prediction_error(reference, candidate) -> dict[str, float | bool]:
    if len(reference) != len(candidate):
        return {
            "equal": False,
            "max_box_error": float("inf"),
            "max_score_error": float("inf"),
        }
    max_box_error = 0.0
    max_score_error = 0.0
    labels_equal = True
    shapes_equal = True
    for expected, actual in zip(reference, candidate):
        shapes_equal &= expected["boxes"].shape == actual["boxes"].shape
        shapes_equal &= expected["scores"].shape == actual["scores"].shape
        if not shapes_equal:
            continue
        if expected["boxes"].numel():
            max_box_error = max(
                max_box_error,
                float((expected["boxes"] - actual["boxes"]).abs().max()),
            )
            max_score_error = max(
                max_score_error,
                float((expected["scores"] - actual["scores"]).abs().max()),
            )
        labels_equal &= torch.equal(expected["labels"], actual["labels"])
    return {
        "equal": bool(
            shapes_equal
            and labels_equal
            and max_box_error == 0.0
            and max_score_error == 0.0
        ),
        "max_box_error": max_box_error,
        "max_score_error": max_score_error,
    }


def _has_nonzero_gradient(gradients) -> bool:
    return any(
        gradient is not None and bool(torch.count_nonzero(gradient))
        for gradient in gradients
    )


def _gradients(loss, parameters, *, retain_graph: bool = True):
    return torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )


def main() -> None:
    args = parse_args()
    if DEVICE.type != "cuda":
        raise RuntimeError("PC-MR + PC-MOC technical test requires CUDA")
    if args.steps < 1 or args.max_batches < args.steps:
        raise ValueError("Require 1 <= steps <= max-batches")
    teacher_path = args.teacher.resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)

    seed_all(args.seed)
    dataset = build_training_datasets(use_patches=False, is_train=True)
    copy_paste_pool = build_copy_paste_pool(dataset)
    if copy_paste_pool:
        dataset.copy_paste_pool = copy_paste_pool
    reliability_threshold = compute_reliability_threshold(dataset)
    sampler = WeightedRandomSampler(
        dataset.get_sample_weights(),
        num_samples=args.max_batches * args.batch_size,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    batches = iter(loader)
    eval_images, _ = _move_batch(*next(batches))

    student = _build(reliability_threshold)
    teacher_checkpoint = torch.load(
        teacher_path, map_location="cpu", weights_only=False)
    teacher = deepcopy(student)
    teacher.load_state_dict(teacher_checkpoint["model"])
    teacher.to(DEVICE).eval()

    student.eval()
    with torch.no_grad():
        predictions_before = student(eval_images)
    attach_pc_micro_rescue_rpn_teacher(student, teacher)
    attach_pc_micro_object_feature_teacher(student, teacher)
    if getattr(student.rpn, "_micro_rescue_teacher") is not teacher:
        raise AssertionError("PC-MR did not reuse the shared teacher")
    if getattr(student.backbone, "_moc_feature_teacher") is not teacher:
        raise AssertionError("PC-MOC did not reuse the shared teacher")
    student.eval()
    with torch.no_grad():
        predictions_after = student(eval_images)
    default_off = _prediction_error(predictions_before, predictions_after)
    if not default_off["equal"]:
        raise AssertionError(f"Attach changed inference: {default_off}")

    teacher_state_duplicated = any(
        "teacher" in key for key in student.state_dict())
    if teacher_state_duplicated:
        raise AssertionError("Frozen teacher leaked into student state_dict")

    optimizer = torch.optim.SGD(
        student.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    student.train()
    if teacher.training:
        raise AssertionError("Unregistered shared teacher left eval mode")
    body_parameters = tuple(
        parameter for parameter in student.backbone.body.parameters()
        if parameter.requires_grad)
    fpn_parameters = tuple(
        parameter for parameter in student.backbone.fpn.parameters()
        if parameter.requires_grad)
    rpn_parameters = tuple(
        parameter for parameter in student.rpn.head.parameters()
        if parameter.requires_grad)
    roi_parameters = tuple(
        parameter for parameter in student.roi_heads.parameters()
        if parameter.requires_grad)
    if set(map(id, fpn_parameters)) & set(map(id, rpn_parameters)):
        raise AssertionError("FPN and RPN PCGrad scopes overlap")

    torch.cuda.reset_peak_memory_stats()
    step_rows = []
    inspected_batches = 0
    while len(step_rows) < args.steps and inspected_batches < args.max_batches:
        try:
            images, targets = next(batches)
        except StopIteration:
            break
        inspected_batches += 1
        images, targets = _move_batch(images, targets)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=True):
            losses = student(images, targets)
            pc_mr_auxiliary = losses["loss_rpn_micro_rescue"]
            pc_moc_auxiliary = losses["loss_fpn_micro_feature"]
            detector = sum(
                value for name, value in losses.items()
                if name not in {
                    "loss_rpn_micro_rescue",
                    "loss_fpn_micro_feature",
                }
            )
            total = detector + pc_mr_auxiliary + pc_moc_auxiliary
        pc_mr_stats = dict(student.rpn._micro_rescue_stats)
        pc_moc_stats = dict(student.backbone._moc_feature_stats)
        if (
            pc_mr_stats.get("selected_gt", 0) == 0
            or pc_moc_stats.get("selected_gt", 0) == 0
        ):
            continue
        if (
            pc_mr_stats.get("micro_gt") != pc_moc_stats.get("micro_gt")
            or pc_mr_stats.get("selected_gt") != pc_moc_stats.get("selected_gt")
        ):
            raise AssertionError("PC-MR and PC-MOC selection diverged")
        if not torch.isfinite(total):
            raise AssertionError(f"Non-finite combined loss: {losses}")

        pc_mr_body = _gradients(pc_mr_auxiliary, body_parameters)
        pc_mr_fpn = _gradients(pc_mr_auxiliary, fpn_parameters)
        pc_mr_rpn = _gradients(pc_mr_auxiliary, rpn_parameters)
        pc_mr_roi = _gradients(pc_mr_auxiliary, roi_parameters)
        if _has_nonzero_gradient(pc_mr_body) or _has_nonzero_gradient(pc_mr_fpn):
            raise AssertionError("PC-MR auxiliary reached the backbone")
        if not _has_nonzero_gradient(pc_mr_rpn):
            raise AssertionError("PC-MR auxiliary missed the RPN head")
        if _has_nonzero_gradient(pc_mr_roi):
            raise AssertionError("PC-MR auxiliary reached the RoI head")

        pc_moc_body = _gradients(pc_moc_auxiliary, body_parameters)
        pc_moc_fpn = _gradients(pc_moc_auxiliary, fpn_parameters)
        pc_moc_rpn = _gradients(pc_moc_auxiliary, rpn_parameters)
        pc_moc_roi = _gradients(pc_moc_auxiliary, roi_parameters)
        if _has_nonzero_gradient(pc_moc_body):
            raise AssertionError("PC-MOC auxiliary reached the backbone body")
        if not _has_nonzero_gradient(pc_moc_fpn):
            raise AssertionError("PC-MOC auxiliary missed the FPN")
        if _has_nonzero_gradient(pc_moc_rpn):
            raise AssertionError("PC-MOC auxiliary reached the RPN head")
        if _has_nonzero_gradient(pc_moc_roi):
            raise AssertionError("PC-MOC auxiliary reached the RoI head")

        pcgrad = _backward_with_disjoint_pcgrad(
            detector,
            (
                ("rpn", pc_mr_auxiliary, rpn_parameters),
                ("fpn", pc_moc_auxiliary, fpn_parameters),
            ),
            scaler,
        )
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        if any(parameter.grad is not None for parameter in teacher.parameters()):
            raise AssertionError("Frozen shared teacher received gradients")
        step_rows.append({
            "step": len(step_rows) + 1,
            "total_loss": float(total.detach()),
            "detector_loss": float(detector.detach()),
            "pc_mr_auxiliary_loss": float(pc_mr_auxiliary.detach()),
            "pc_moc_auxiliary_loss": float(pc_moc_auxiliary.detach()),
            "pc_mr_selection": pc_mr_stats,
            "pc_moc_selection": pc_moc_stats,
            "pcgrad": pcgrad,
        })

    if len(step_rows) != args.steps:
        raise AssertionError(
            f"Only {len(step_rows)}/{args.steps} valid combined steps")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.out_json.with_suffix(".pt")
    torch.save({
        "model": student.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
    }, checkpoint_path)
    reloaded = _build(reliability_threshold)
    reload_payload = torch.load(
        checkpoint_path, map_location=DEVICE, weights_only=False)
    reloaded.load_state_dict(reload_payload["model"])
    reload_optimizer = torch.optim.SGD(
        reloaded.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    reload_optimizer.load_state_dict(reload_payload["optimizer"])
    reload_scaler = torch.amp.GradScaler("cuda", enabled=True)
    reload_scaler.load_state_dict(reload_payload["scaler"])

    student.eval()
    reloaded.eval()
    with torch.no_grad():
        trained_predictions = student(eval_images)
        reloaded_predictions = reloaded(eval_images)
    reload_equivalence = _prediction_error(
        trained_predictions, reloaded_predictions)
    if not reload_equivalence["equal"]:
        raise AssertionError(
            f"Reload changed inference: {reload_equivalence}")

    artifact = {
        "method": "PC-MR-RPN+PC-MOC-FD",
        "seed": args.seed,
        "teacher": str(teacher_path),
        "teacher_epoch": teacher_checkpoint.get("epoch"),
        "teacher_model_source": teacher_checkpoint.get("model_source"),
        "shared_teacher_object": True,
        "batch_size": args.batch_size,
        "steps": step_rows,
        "inspected_batches": inspected_batches,
        "default_off_inference_equivalence": default_off,
        "reload_inference_equivalence": reload_equivalence,
        "teacher_state_duplicated": teacher_state_duplicated,
        "teacher_gradient_parameters": sum(
            parameter.grad is not None for parameter in teacher.parameters()),
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "checkpoint": str(checkpoint_path),
        "pass": True,
    }
    args.out_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
