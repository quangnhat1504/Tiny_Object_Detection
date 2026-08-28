"""Audit a teacher-gated micro-object RPN auxiliary loss without updates."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.models.detection.rpn import concat_box_prediction_layers
from torchvision.ops import box_iou

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
from common.model import build_model
from scripts.audit_cbl_cross_scale_gradient_conflict import (
    _gradient_pair_stats,
    _quantile,
)


DEFAULT_TEACHER = (
    ROOT
    / ".runtime/kaggle/cbl_iterative_train_fair20/output/tod_output/runs"
    / "sa_alw_full__cbl__irtw0.5ir1s0.3__la_loss__seed42__cbl_iterative_train_fair20"
    / "best.pt"
)
DEFAULT_OUTPUT = ROOT / "runs/micro_rescue_rpn_gradient_audit_seed42.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit teacher-gated micro-rescue RPN gradients")
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--proposal-top-n", type=int, default=300)
    parser.add_argument("--micro-cutoff-px", type=float, default=8.0)
    parser.add_argument("--teacher-iou-floor", type=float, default=0.50)
    parser.add_argument("--advantage-margin", type=float, default=0.02)
    parser.add_argument("--loss-weight", type=float, default=0.05)
    parser.add_argument("--teacher-min-size", type=int, default=960)
    parser.add_argument("--teacher-max-size", type=int, default=1200)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


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


def _rpn_snapshot(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict],
    min_size: int,
    max_size: int,
    require_grad: bool,
) -> dict:
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size
    context = torch.enable_grad() if require_grad else torch.no_grad()
    with context:
        image_list, transformed_targets = model.transform(images, targets)
        features = model.backbone(image_list.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        feature_list = list(features.values())
        objectness_list, bbox_delta_list = model.rpn.head(feature_list)
        anchors = model.rpn.anchor_generator(image_list, feature_list)
        num_images = len(anchors)
        num_anchors_per_level = [
            score[0].shape[0] * score[0].shape[1] * score[0].shape[2]
            for score in objectness_list
        ]
        objectness, bbox_deltas = concat_box_prediction_layers(
            objectness_list, bbox_delta_list)
        objectness_by_image = objectness.reshape(num_images, -1)
        bbox_deltas_by_image = bbox_deltas.reshape(num_images, -1, 4)
        decoded = model.rpn.box_coder.decode(
            bbox_deltas.detach(), anchors).reshape(num_images, -1, 4)
        was_training = model.rpn.training
        model.rpn.eval()
        try:
            filtered, _ = model.rpn.filter_proposals(
                decoded,
                objectness,
                image_list.image_sizes,
                num_anchors_per_level,
            )
            pre_nms_indices = model.rpn._get_top_n_idx(
                objectness_by_image.detach(), num_anchors_per_level)
        finally:
            model.rpn.train(was_training)
    return {
        "targets": transformed_targets,
        "anchors": anchors,
        "objectness": objectness_by_image,
        "bbox_deltas": bbox_deltas_by_image,
        "decoded": decoded,
        "filtered": filtered,
        "pre_nms_indices": pre_nms_indices,
    }


def _best_filtered_iou(
    gt_boxes: torch.Tensor,
    proposals: torch.Tensor,
    top_n: int,
) -> torch.Tensor:
    proposals = proposals[:top_n]
    if len(gt_boxes) == 0 or len(proposals) == 0:
        return torch.zeros(len(gt_boxes), device=gt_boxes.device)
    return box_iou(gt_boxes, proposals).max(dim=1).values


def _micro_rescue_loss(
    student: torch.nn.Module,
    student_snapshot: dict,
    teacher_snapshot: dict,
    original_targets: list[dict],
    proposal_top_n: int,
    micro_cutoff_px: float,
    teacher_iou_floor: float,
    advantage_margin: float,
    loss_weight: float,
) -> tuple[dict[str, torch.Tensor], dict]:
    selected_objectness = []
    selected_deltas = []
    selected_targets = []
    selected_weights = []
    teacher_ious = []
    student_ious = []
    micro_gt_count = 0

    for image_index, original in enumerate(original_targets):
        original_boxes = original["boxes"]
        micro_mask = (
            ((original_boxes[:, 2] - original_boxes[:, 0]).clamp(min=0)
             * (original_boxes[:, 3] - original_boxes[:, 1]).clamp(min=0))
            .sqrt()
            < micro_cutoff_px
        )
        micro_gt_count += int(micro_mask.sum().item())
        if not micro_mask.any():
            continue

        student_gt = student_snapshot["targets"][image_index]["boxes"]
        teacher_gt = teacher_snapshot["targets"][image_index]["boxes"]
        if len(student_gt) != len(teacher_gt) or len(student_gt) != len(original_boxes):
            raise AssertionError("Cross-scale GT order/count changed")
        student_best = _best_filtered_iou(
            student_gt,
            student_snapshot["filtered"][image_index],
            proposal_top_n,
        )
        teacher_best = _best_filtered_iou(
            teacher_gt,
            teacher_snapshot["filtered"][image_index],
            proposal_top_n,
        )
        advantage = teacher_best - student_best
        selected_gt_mask = (
            micro_mask
            & (teacher_best >= teacher_iou_floor)
            & (advantage >= advantage_margin)
        )
        selected_gt_indices = torch.where(selected_gt_mask)[0]
        if selected_gt_indices.numel() == 0:
            continue

        candidate_indices = student_snapshot["pre_nms_indices"][image_index]
        candidate_boxes = student_snapshot["decoded"][
            image_index, candidate_indices]
        candidate_iou = box_iou(
            student_gt[selected_gt_indices], candidate_boxes)
        best_candidate_positions = candidate_iou.argmax(dim=1)
        best_anchor_indices = candidate_indices[best_candidate_positions]
        selected_anchors = student_snapshot["anchors"][image_index][
            best_anchor_indices]
        gt_targets = student_gt[selected_gt_indices]
        encoded_targets = student.rpn.box_coder.encode_single(
            gt_targets, selected_anchors)

        selected_objectness.append(
            student_snapshot["objectness"][
                image_index, best_anchor_indices])
        selected_deltas.append(
            student_snapshot["bbox_deltas"][
                image_index, best_anchor_indices])
        selected_targets.append(encoded_targets)
        selected_weights.append(advantage[selected_gt_indices].detach())
        teacher_ious.append(teacher_best[selected_gt_indices].detach())
        student_ious.append(student_best[selected_gt_indices].detach())

    if not selected_weights:
        zero = student_snapshot["objectness"].sum() * 0
        return {"joint": zero, "objectness": zero, "regression": zero}, {
            "micro_gt": micro_gt_count,
            "selected_gt": 0,
            "weight_sum": 0.0,
            "teacher_iou_mean": 0.0,
            "student_iou_mean": 0.0,
            "raw_objectness_loss": 0.0,
            "raw_regression_loss": 0.0,
        }

    objectness = torch.cat(selected_objectness)
    predicted_deltas = torch.cat(selected_deltas)
    regression_targets = torch.cat(selected_targets)
    weights = torch.cat(selected_weights).float()
    weight_sum = weights.sum().clamp_min(1e-12)
    objectness_per_gt = F.binary_cross_entropy_with_logits(
        objectness.float(),
        torch.ones_like(objectness, dtype=torch.float32),
        reduction="none",
    )
    regression_per_gt = F.smooth_l1_loss(
        predicted_deltas.float(),
        regression_targets.float(),
        beta=1 / 9,
        reduction="none",
    ).sum(dim=1)
    raw_objectness_loss = (weights * objectness_per_gt).sum() / weight_sum
    raw_regression_loss = (weights * regression_per_gt).sum() / weight_sum
    losses = {
        "objectness": loss_weight * raw_objectness_loss,
        "regression": loss_weight * raw_regression_loss,
    }
    losses["joint"] = losses["objectness"] + losses["regression"]
    return losses, {
        "micro_gt": micro_gt_count,
        "selected_gt": int(weights.numel()),
        "weight_sum": float(weights.sum().item()),
        "teacher_iou_mean": float(torch.cat(teacher_ious).mean().item()),
        "student_iou_mean": float(torch.cat(student_ious).mean().item()),
        "raw_objectness_loss": float(raw_objectness_loss.detach().item()),
        "raw_regression_loss": float(raw_regression_loss.detach().item()),
    }


def _summarize(rows: list[dict]) -> dict:
    valid = [row for row in rows if row["gradient"]["finite_nonzero"]]
    cosines = [row["gradient"]["cosine"] for row in valid]
    ratios = [row["gradient"]["norm_ratio"] for row in valid]
    micro_gt = sum(row["micro_gt"] for row in rows)
    selected_gt = sum(row["selected_gt"] for row in rows)
    summary = {
        "batches": len(rows),
        "valid_gradient_batches": len(valid),
        "conflict_batches": sum(
            bool(row["gradient"]["conflict"]) for row in valid),
        "conflict_rate": (
            sum(bool(row["gradient"]["conflict"]) for row in valid)
            / len(valid) if valid else float("nan")),
        "cosine_mean": statistics.fmean(cosines) if cosines else float("nan"),
        "cosine_median": (
            statistics.median(cosines) if cosines else float("nan")),
        "cosine_q1": _quantile(cosines, 0.25),
        "cosine_q3": _quantile(cosines, 0.75),
        "norm_ratio_mean": (
            statistics.fmean(ratios) if ratios else float("nan")),
        "norm_ratio_median": (
            statistics.median(ratios) if ratios else float("nan")),
        "micro_gt": micro_gt,
        "selected_gt": selected_gt,
        "selection_coverage": (
            selected_gt / micro_gt if micro_gt else float("nan")),
        "weight_sum": sum(row["weight_sum"] for row in rows),
    }
    component_summaries = {}
    for component in ("objectness", "regression"):
        component_valid = [
            row["component_gradients"][component]
            for row in rows
            if row["component_gradients"][component]["finite_nonzero"]
        ]
        component_cosines = [item["cosine"] for item in component_valid]
        component_ratios = [item["norm_ratio"] for item in component_valid]
        component_summaries[component] = {
            "valid_gradient_batches": len(component_valid),
            "conflict_batches": sum(
                bool(item["conflict"]) for item in component_valid),
            "conflict_rate": (
                sum(bool(item["conflict"]) for item in component_valid)
                / len(component_valid) if component_valid else float("nan")),
            "cosine_mean": (
                statistics.fmean(component_cosines)
                if component_cosines else float("nan")),
            "cosine_median": (
                statistics.median(component_cosines)
                if component_cosines else float("nan")),
            "norm_ratio_mean": (
                statistics.fmean(component_ratios)
                if component_ratios else float("nan")),
            "norm_ratio_median": (
                statistics.median(component_ratios)
                if component_ratios else float("nan")),
        }
    summary["component_summaries"] = component_summaries
    parameter_group_summaries = {}
    for component in ("objectness", "regression"):
        parameter_group_summaries[component] = {}
        group_names = (
            rows[0]["parameter_group_gradients"][component].keys()
            if rows else ())
        for group_name in group_names:
            group_valid = [
                row["parameter_group_gradients"][component][group_name]
                for row in rows
                if row["parameter_group_gradients"][component][group_name][
                    "finite_nonzero"]
            ]
            group_cosines = [item["cosine"] for item in group_valid]
            group_ratios = [item["norm_ratio"] for item in group_valid]
            parameter_group_summaries[component][group_name] = {
                "valid_gradient_batches": len(group_valid),
                "conflict_batches": sum(
                    bool(item["conflict"]) for item in group_valid),
                "conflict_rate": (
                    sum(bool(item["conflict"]) for item in group_valid)
                    / len(group_valid) if group_valid else float("nan")),
                "cosine_mean": (
                    statistics.fmean(group_cosines)
                    if group_cosines else float("nan")),
                "cosine_median": (
                    statistics.median(group_cosines)
                    if group_cosines else float("nan")),
                "norm_ratio_mean": (
                    statistics.fmean(group_ratios)
                    if group_ratios else float("nan")),
                "norm_ratio_median": (
                    statistics.median(group_ratios)
                    if group_ratios else float("nan")),
            }
    summary["parameter_group_summaries"] = parameter_group_summaries
    regression_rows = [
        row["component_gradients"]["regression"]
        for row in rows
        if row["component_gradients"]["regression"]["finite_nonzero"]
    ]
    projected_cosines = [
        max(0.0, float(item["cosine"])) for item in regression_rows]
    projected_ratios = [
        float(item["norm_ratio"])
        * (
            math.sqrt(max(0.0, 1.0 - float(item["cosine"]) ** 2))
            if float(item["cosine"]) < 0 else 1.0
        )
        for item in regression_rows
    ]
    raw_conflicts = sum(
        bool(item["conflict"]) for item in regression_rows)
    summary["pc_regression"] = {
        "valid_gradient_batches": len(regression_rows),
        "valid_batch_rate": (
            len(regression_rows) / len(rows) if rows else float("nan")),
        "raw_conflict_batches": raw_conflicts,
        "raw_conflict_rate": (
            raw_conflicts / len(regression_rows)
            if regression_rows else float("nan")),
        "projected_conflict_rate": 0.0 if regression_rows else float("nan"),
        "projected_cosine_mean": (
            statistics.fmean(projected_cosines)
            if projected_cosines else float("nan")),
        "projected_norm_ratio_mean": (
            statistics.fmean(projected_ratios)
            if projected_ratios else float("nan")),
    }
    return summary


def _write_artifact(
    path: Path,
    protocol: dict,
    rows: list[dict],
    complete: bool,
    elapsed_seconds: float,
) -> None:
    summary = _summarize(rows)
    summary["gate_conflict_rate_max"] = 0.05
    summary["gate_norm_ratio_range"] = [0.02, 0.15]
    summary["gate_selection_coverage_min"] = 0.10
    summary["gate_pass"] = bool(
        complete
        and summary["valid_gradient_batches"] >= protocol["batches"]
        and summary["selection_coverage"] >= 0.10
        and summary["conflict_rate"] <= 0.05
        and summary["cosine_mean"] > 0
        and 0.02 <= summary["norm_ratio_mean"] <= 0.15
    )
    pc_regression = summary["pc_regression"]
    summary["pc_regression_gate"] = {
        "valid_batch_rate_min": 0.50,
        "selection_coverage_min": 0.10,
        "raw_conflict_rate_min": 0.10,
        "projected_conflict_rate_max": 0.0,
        "projected_cosine_min": 0.0,
        "projected_norm_ratio_range": [0.02, 0.15],
        "gate_pass": bool(
            complete
            and protocol["loss_weight"] == 0.005
            and pc_regression["valid_batch_rate"] >= 0.50
            and summary["selection_coverage"] >= 0.10
            and pc_regression["raw_conflict_rate"] >= 0.10
            and pc_regression["projected_conflict_rate"] <= 0.0
            and pc_regression["projected_cosine_mean"] > 0.0
            and 0.02 <= pc_regression["projected_norm_ratio_mean"] <= 0.15
        ),
    }
    artifact = {
        "complete": complete,
        "protocol": protocol,
        "elapsed_seconds": elapsed_seconds,
        "summary": summary,
        "batches": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.batches < 1 or args.batch_size < 1 or args.save_every < 1:
        raise ValueError("batches, batch-size, and save-every must be positive")
    if args.proposal_top_n < 1 or args.micro_cutoff_px <= 0:
        raise ValueError("proposal-top-n and micro-cutoff-px must be positive")
    if not 0 <= args.teacher_iou_floor <= 1:
        raise ValueError("teacher-iou-floor must be in [0, 1]")
    if args.advantage_margin < 0 or args.loss_weight <= 0:
        raise ValueError("advantage-margin must be non-negative and loss-weight positive")
    if min(args.teacher_min_size, args.teacher_max_size) < 1:
        raise ValueError("teacher transform sizes must be positive")
    teacher_path = args.teacher.resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)

    seed_all(args.seed)
    dataset = build_training_datasets(use_patches=False, is_train=True)
    copy_paste_pool = build_copy_paste_pool(dataset)
    if copy_paste_pool:
        dataset.copy_paste_pool = copy_paste_pool
    reliability_threshold = compute_reliability_threshold(dataset)
    generator = torch.Generator().manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        dataset.get_sample_weights(),
        num_samples=args.batches * args.batch_size,
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=DEVICE.type == "cuda",
        drop_last=True,
    )

    student = build_model(
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
    checkpoint = torch.load(
        teacher_path, map_location="cpu", weights_only=False)
    teacher = deepcopy(student)
    teacher.load_state_dict(checkpoint["model"])
    teacher.to(DEVICE).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.train()
    shared_named_parameters = tuple(
        (name, parameter)
        for name, parameter in student.rpn.head.named_parameters()
        if parameter.requires_grad)
    shared_parameters = tuple(
        parameter for _, parameter in shared_named_parameters)
    if not shared_parameters:
        raise RuntimeError("RPN head has no trainable parameters")
    parameter_group_indices = {
        "shared_conv": tuple(
            index for index, (name, _) in enumerate(shared_named_parameters)
            if name.startswith("conv.")),
        "cls_logits": tuple(
            index for index, (name, _) in enumerate(shared_named_parameters)
            if name.startswith("cls_logits.")),
        "bbox_pred": tuple(
            index for index, (name, _) in enumerate(shared_named_parameters)
            if name.startswith("bbox_pred.")),
    }
    if any(not indices for indices in parameter_group_indices.values()):
        raise RuntimeError(
            f"Unexpected RPN head parameter groups: "
            f"{[name for name, _ in shared_named_parameters]}")

    amp_enabled = DEVICE.type == "cuda" and not args.no_amp
    protocol = {
        "seed": args.seed,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "weighted_sampler": True,
        "copy_paste": bool(copy_paste_pool),
        "student_initialization": "torchvision FasterRCNN ResNet50-FPN DEFAULT",
        "student_scale": [640, 800],
        "teacher_path": str(teacher_path),
        "teacher_epoch": checkpoint.get("epoch"),
        "teacher_model_source": checkpoint.get("model_source"),
        "teacher_scale": [args.teacher_min_size, args.teacher_max_size],
        "proposal_top_n": args.proposal_top_n,
        "micro_cutoff_px": args.micro_cutoff_px,
        "teacher_iou_floor": args.teacher_iou_floor,
        "advantage_margin": args.advantage_margin,
        "loss_weight": args.loss_weight,
        "shared_parameters": "student RPN head",
        "parameter_groups": {
            key: [shared_named_parameters[index][0] for index in indices]
            for key, indices in parameter_group_indices.items()
        },
        "reference_loss": "loss_objectness + loss_rpn_box_reg",
        "auxiliary_target": "exact GT objectness and encoded box delta",
        "amp": amp_enabled,
        "optimizer_updates": 0,
    }
    del checkpoint

    rows = []
    started = time.time()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images, targets = _move_batch(images, targets)
        student.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            detector_losses = student(images, targets)
            reference_loss = (
                detector_losses["loss_objectness"]
                + detector_losses["loss_rpn_box_reg"])
            student_snapshot = _rpn_snapshot(
                student, images, targets, 640, 800, require_grad=True)
            teacher_snapshot = _rpn_snapshot(
                teacher,
                images,
                targets,
                args.teacher_min_size,
                args.teacher_max_size,
                require_grad=False,
            )
            auxiliary_losses, telemetry = _micro_rescue_loss(
                student,
                student_snapshot,
                teacher_snapshot,
                targets,
                args.proposal_top_n,
                args.micro_cutoff_px,
                args.teacher_iou_floor,
                args.advantage_margin,
                args.loss_weight,
            )
        reference_gradients = torch.autograd.grad(
            reference_loss,
            shared_parameters,
            allow_unused=True,
        )
        component_gradients = {}
        parameter_group_gradients = {}
        for component_index, component in enumerate(
                ("joint", "objectness", "regression")):
            auxiliary_gradients = torch.autograd.grad(
                auxiliary_losses[component],
                shared_parameters,
                retain_graph=component_index < 2,
                allow_unused=True,
            )
            component_gradients[component] = _gradient_pair_stats(
                reference_gradients, auxiliary_gradients)
            if component != "joint":
                parameter_group_gradients[component] = {
                    group_name: _gradient_pair_stats(
                        tuple(reference_gradients[index] for index in indices),
                        tuple(auxiliary_gradients[index] for index in indices),
                    )
                    for group_name, indices in parameter_group_indices.items()
                }
        row = {
            "batch": batch_index,
            "reference_loss": float(reference_loss.detach().item()),
            "auxiliary_loss": float(auxiliary_losses["joint"].detach().item()),
            "auxiliary_losses": {
                key: float(value.detach().item())
                for key, value in auxiliary_losses.items()
            },
            **telemetry,
            "gradient": component_gradients["joint"],
            "component_gradients": component_gradients,
            "parameter_group_gradients": parameter_group_gradients,
        }
        rows.append(row)
        assert all(parameter.grad is None for parameter in teacher.parameters())
        if batch_index % args.save_every == 0 or batch_index == args.batches:
            elapsed = time.time() - started
            _write_artifact(
                args.out_json,
                protocol,
                rows,
                complete=batch_index == args.batches,
                elapsed_seconds=elapsed,
            )
            summary = _summarize(rows)
            print(
                f"[{batch_index:03d}/{args.batches}] "
                f"valid={summary['valid_gradient_batches']}, "
                f"coverage={summary['selection_coverage']:.3f}, "
                f"conflict={summary['conflict_rate']:.3f}, "
                f"cos={summary['cosine_mean']:.4f}, "
                f"ratio={summary['norm_ratio_mean']:.4f}, "
                f"elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

    artifact = json.loads(args.out_json.read_text(encoding="utf-8"))
    artifact["peak_allocated_gib"] = (
        torch.cuda.max_memory_allocated() / 1024**3
        if DEVICE.type == "cuda" else 0.0)
    args.out_json.write_text(
        json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))
    print(f"Saved audit: {args.out_json}")


if __name__ == "__main__":
    main()
