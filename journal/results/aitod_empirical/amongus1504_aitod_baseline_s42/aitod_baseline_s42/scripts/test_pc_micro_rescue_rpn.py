"""Real-data CUDA contract test for PC-MR-RPN."""

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
from common.model import attach_pc_micro_rescue_rpn_teacher, build_model
from common.train_utils import _backward_with_pcgrad


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
        default=ROOT / "runs/pc_micro_rescue_rpn_technical_smoke_seed42.json",
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
        return {"equal": False, "max_box_error": float("inf"),
                "max_score_error": float("inf")}
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
                float((expected["boxes"] - actual["boxes"]).abs().max().item()),
            )
            max_score_error = max(
                max_score_error,
                float((expected["scores"] - actual["scores"]).abs().max().item()),
            )
        labels_equal &= torch.equal(expected["labels"], actual["labels"])
    return {
        "equal": bool(
            shapes_equal and labels_equal
            and max_box_error == 0.0 and max_score_error == 0.0),
        "max_box_error": max_box_error,
        "max_score_error": max_score_error,
    }


def main() -> None:
    args = parse_args()
    if DEVICE.type != "cuda":
        raise RuntimeError("PC-MR-RPN technical test requires CUDA")
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
    eval_images, eval_targets = _move_batch(*next(batches))

    student = _build(reliability_threshold)
    teacher_checkpoint = torch.load(
        teacher_path, map_location="cpu", weights_only=False)
    teacher = deepcopy(student)
    teacher.load_state_dict(teacher_checkpoint["model"])
    teacher.to(DEVICE)

    student.eval()
    with torch.no_grad():
        predictions_before = student(eval_images)
    attach_pc_micro_rescue_rpn_teacher(student, teacher)
    student.eval()
    with torch.no_grad():
        predictions_after = student(eval_images)
    default_off = _prediction_error(predictions_before, predictions_after)
    if not default_off["equal"]:
        raise AssertionError(f"Attach changed inference: {default_off}")

    teacher_state_duplicated = any(
        "micro_rescue_teacher" in key for key in student.state_dict())
    if teacher_state_duplicated:
        raise AssertionError("Frozen teacher leaked into student state_dict")

    optimizer = torch.optim.SGD(
        student.parameters(), lr=1e-4, momentum=0.9, weight_decay=5e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    student.train()
    torch.cuda.reset_peak_memory_stats()
    step_rows = []
    inspected_batches = 0
    backbone_parameters = tuple(
        parameter for parameter in student.backbone.parameters()
        if parameter.requires_grad)
    rpn_parameters = tuple(
        parameter for parameter in student.rpn.head.parameters()
        if parameter.requires_grad)
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
            auxiliary = losses["loss_rpn_micro_rescue"]
            detector = sum(
                value for name, value in losses.items()
                if name != "loss_rpn_micro_rescue")
            total = detector + auxiliary
        stats = dict(student.rpn._micro_rescue_stats)
        if stats.get("selected_gt", 0) == 0:
            continue
        if not torch.isfinite(total):
            raise AssertionError(f"Non-finite PC-MR-RPN loss: {losses}")

        backbone_aux = torch.autograd.grad(
            scaler.scale(auxiliary),
            backbone_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        if any(
            gradient is not None and bool(torch.count_nonzero(gradient))
            for gradient in backbone_aux
        ):
            raise AssertionError("PC-MR-RPN auxiliary reached the backbone")
        rpn_aux = torch.autograd.grad(
            scaler.scale(auxiliary),
            rpn_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        if not any(
            gradient is not None and bool(torch.count_nonzero(gradient))
            for gradient in rpn_aux
        ):
            raise AssertionError("PC-MR-RPN auxiliary missed the RPN head")

        pcgrad = _backward_with_pcgrad(
            detector, auxiliary, rpn_parameters, scaler)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        if any(parameter.grad is not None for parameter in teacher.parameters()):
            raise AssertionError("Frozen PC-MR-RPN teacher received gradients")
        step_rows.append({
            "step": len(step_rows) + 1,
            "total_loss": float(total.detach().item()),
            "detector_loss": float(detector.detach().item()),
            "auxiliary_loss": float(auxiliary.detach().item()),
            "selection": stats,
            "pcgrad": pcgrad,
        })

    if len(step_rows) != args.steps:
        raise AssertionError(
            f"Only {len(step_rows)}/{args.steps} valid PC-MR-RPN steps")

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
        "method": "PC-MR-RPN",
        "seed": args.seed,
        "teacher": str(teacher_path),
        "teacher_epoch": teacher_checkpoint.get("epoch"),
        "teacher_model_source": teacher_checkpoint.get("model_source"),
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
