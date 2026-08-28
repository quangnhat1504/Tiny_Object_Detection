"""Audit teacher-bounded micro-object FPN distillation without updates."""

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
from common.model import _micro_high_frequency_feature_loss, build_model
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
DEFAULT_OUTPUT = ROOT / "runs/moc_fd_fpn_gradient_probe_seed42.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure detector/FPN gradients for teacher-bounded micro-object "
            "feature distillation; no optimizer step is performed."
        )
    )
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--proposal-top-n", type=int, default=300)
    parser.add_argument("--micro-cutoff-px", type=float, default=8.0)
    parser.add_argument("--teacher-iou-floor", type=float, default=0.50)
    parser.add_argument("--advantage-margin", type=float, default=0.02)
    parser.add_argument("--loss-weight", type=float, default=0.01)
    parser.add_argument(
        "--feature-target",
        choices=(
            "cosine",
            "spatial_dependency",
            "high_frequency",
            "high_frequency_energy",
        ),
        default="cosine",
        help=(
            "Feature target: pointwise channel cosine (PC-MOC-FD) or "
            "within-RoI spatial-relation distributions (PC-MSDD), or "
            "teacher-energy-weighted local residuals (PC-MHFD), or a "
            "channel-invariant residual-energy map (PC-MHED)."
        ),
    )
    parser.add_argument(
        "--spatial-temperature",
        type=float,
        default=0.20,
        help="Softmax temperature for the PC-MSDD spatial relation target",
    )
    parser.add_argument("--student-min-size", type=int, default=640)
    parser.add_argument("--student-max-size", type=int, default=800)
    parser.add_argument("--teacher-min-size", type=int, default=960)
    parser.add_argument("--teacher-max-size", type=int, default=1200)
    parser.add_argument(
        "--full-gate",
        action="store_true",
        help="Apply the frozen 200-batch PC-MOC-FD continuation gate",
    )
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


def _as_ordered_dict(features) -> OrderedDict[str, torch.Tensor]:
    if isinstance(features, torch.Tensor):
        return OrderedDict([("0", features)])
    return OrderedDict(features)


def _filtered_rpn_proposals(
    model: torch.nn.Module,
    image_list,
    features: OrderedDict[str, torch.Tensor],
) -> list[torch.Tensor]:
    with torch.no_grad():
        feature_list = [feature.detach() for feature in features.values()]
        objectness_list, bbox_delta_list = model.rpn.head(feature_list)
        anchors = model.rpn.anchor_generator(image_list, feature_list)
        num_images = len(anchors)
        num_anchors_per_level = [
            score[0].shape[0] * score[0].shape[1] * score[0].shape[2]
            for score in objectness_list
        ]
        objectness, bbox_deltas = concat_box_prediction_layers(
            objectness_list, bbox_delta_list)
        decoded = model.rpn.box_coder.decode(
            bbox_deltas, anchors).reshape(num_images, -1, 4)
        was_training = model.rpn.training
        model.rpn.eval()
        try:
            filtered, _ = model.rpn.filter_proposals(
                decoded,
                objectness,
                image_list.image_sizes,
                num_anchors_per_level,
            )
        finally:
            model.rpn.train(was_training)
    return filtered


def _fpn_snapshot(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict],
    min_size: int,
    max_size: int,
    require_fpn_grad: bool,
) -> dict:
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size
    image_list, transformed_targets = model.transform(images, targets)
    with torch.no_grad():
        body_features = _as_ordered_dict(
            model.backbone.body(image_list.tensors))
        body_features = OrderedDict(
            (name, feature.detach())
            for name, feature in body_features.items()
        )
    context = torch.enable_grad() if require_fpn_grad else torch.no_grad()
    with context:
        features = _as_ordered_dict(model.backbone.fpn(body_features))
        pooled = model.roi_heads.box_roi_pool(
            features,
            [target["boxes"] for target in transformed_targets],
            image_list.image_sizes,
        )
    proposals = _filtered_rpn_proposals(model, image_list, features)
    counts = [len(target["boxes"]) for target in transformed_targets]
    return {
        "targets": transformed_targets,
        "pooled": pooled.split(counts, dim=0),
        "proposals": proposals,
    }


def _best_iou(
    gt_boxes: torch.Tensor,
    proposals: torch.Tensor,
    top_n: int,
) -> torch.Tensor:
    proposals = proposals[:top_n]
    if len(gt_boxes) == 0 or len(proposals) == 0:
        return torch.zeros(len(gt_boxes), device=gt_boxes.device)
    return box_iou(gt_boxes, proposals).max(dim=1).values


def _feature_loss(
    student_snapshot: dict,
    teacher_snapshot: dict,
    original_targets: list[dict],
    proposal_top_n: int,
    micro_cutoff_px: float,
    teacher_iou_floor: float,
    advantage_margin: float,
    loss_weight: float,
    feature_target: str = "cosine",
    spatial_temperature: float = 0.20,
) -> tuple[torch.Tensor, dict]:
    per_roi_losses = []
    selected_weights = []
    teacher_ious = []
    student_ious = []
    micro_gt_count = 0

    for image_index, original in enumerate(original_targets):
        boxes = original["boxes"]
        sqrt_area = (
            (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
            * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        ).sqrt()
        micro_mask = sqrt_area < micro_cutoff_px
        micro_gt_count += int(micro_mask.sum().item())
        if not micro_mask.any():
            continue

        student_gt = student_snapshot["targets"][image_index]["boxes"]
        teacher_gt = teacher_snapshot["targets"][image_index]["boxes"]
        if len(student_gt) != len(boxes) or len(teacher_gt) != len(boxes):
            raise AssertionError("Cross-scale transform changed GT order/count")
        student_best = _best_iou(
            student_gt,
            student_snapshot["proposals"][image_index],
            proposal_top_n,
        )
        teacher_best = _best_iou(
            teacher_gt,
            teacher_snapshot["proposals"][image_index],
            proposal_top_n,
        )
        advantage = teacher_best - student_best
        selected = (
            micro_mask
            & (teacher_best >= teacher_iou_floor)
            & (advantage >= advantage_margin)
        )
        if not selected.any():
            continue

        student_features = student_snapshot["pooled"][image_index][selected]
        teacher_features = teacher_snapshot["pooled"][image_index][selected]
        if feature_target == "cosine":
            student_features = F.normalize(student_features.float(), dim=1)
            teacher_features = F.normalize(
                teacher_features.float().detach(), dim=1)
            cosine = (student_features * teacher_features).sum(dim=1)
            roi_losses = (1.0 - cosine).mean(dim=(1, 2))
        elif feature_target == "spatial_dependency":
            roi_losses = _spatial_dependency_loss(
                student_features,
                teacher_features,
                temperature=spatial_temperature,
            )
        elif feature_target == "high_frequency":
            roi_losses = _micro_high_frequency_feature_loss(
                student_features,
                teacher_features,
            )
        elif feature_target == "high_frequency_energy":
            roi_losses = _micro_high_frequency_energy_loss(
                student_features,
                teacher_features,
            )
        else:
            raise ValueError(f"Unknown feature target: {feature_target}")
        per_roi_losses.append(roi_losses)
        selected_weights.append(advantage[selected].detach().float())
        teacher_ious.append(teacher_best[selected].detach())
        student_ious.append(student_best[selected].detach())

    if not selected_weights:
        zero = sum(
            pooled.sum() * 0
            for pooled in student_snapshot["pooled"]
        )
        return zero, {
            "micro_gt": micro_gt_count,
            "selected_gt": 0,
            "weight_sum": 0.0,
            "raw_feature_loss": 0.0,
            "teacher_iou_mean": 0.0,
            "student_iou_mean": 0.0,
        }

    losses = torch.cat(per_roi_losses)
    weights = torch.cat(selected_weights)
    raw_loss = (weights * losses).sum() / weights.sum().clamp_min(1e-12)
    return loss_weight * raw_loss, {
        "micro_gt": micro_gt_count,
        "selected_gt": int(weights.numel()),
        "weight_sum": float(weights.sum().item()),
        "raw_feature_loss": float(raw_loss.detach().item()),
        "teacher_iou_mean": float(torch.cat(teacher_ious).mean().item()),
        "student_iou_mean": float(torch.cat(student_ious).mean().item()),
    }


def _spatial_dependency_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Match each RoI pixel's distribution of relations to all other pixels."""
    if temperature <= 0:
        raise ValueError("Spatial dependency temperature must be positive")
    if student_features.shape != teacher_features.shape:
        raise ValueError("Student/teacher RoI feature shapes must match")
    if student_features.ndim != 4:
        raise ValueError("Expected RoI features shaped [N, C, H, W]")

    student = F.normalize(student_features.float().flatten(2), dim=1)
    teacher = F.normalize(
        teacher_features.float().detach().flatten(2), dim=1)
    student_affinity = torch.bmm(student.transpose(1, 2), student)
    teacher_affinity = torch.bmm(teacher.transpose(1, 2), teacher)

    spatial_positions = student_affinity.shape[-1]
    diagonal = torch.eye(
        spatial_positions,
        dtype=torch.bool,
        device=student_affinity.device,
    ).unsqueeze(0)
    student_logits = (student_affinity / temperature).masked_fill(
        diagonal, torch.finfo(student_affinity.dtype).min)
    teacher_logits = (teacher_affinity / temperature).masked_fill(
        diagonal, torch.finfo(teacher_affinity.dtype).min)
    student_log_prob = F.log_softmax(student_logits, dim=-1)
    teacher_prob = F.softmax(teacher_logits, dim=-1)
    per_position = F.kl_div(
        student_log_prob,
        teacher_prob,
        reduction="none",
    ).sum(dim=-1)
    return per_position.mean(dim=-1)


def _micro_high_frequency_energy_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
) -> torch.Tensor:
    """Match the spatial layout of channel-invariant high-frequency energy."""
    if student_features.shape != teacher_features.shape:
        raise ValueError("Student/teacher RoI feature shapes must match")
    if student_features.ndim != 4:
        raise ValueError("Expected RoI features shaped [N, C, H, W]")

    student = F.normalize(student_features.float(), dim=1)
    teacher = F.normalize(teacher_features.float().detach(), dim=1)
    student_high = student - F.avg_pool2d(
        student, kernel_size=3, stride=1, padding=1,
        count_include_pad=False)
    teacher_high = teacher - F.avg_pool2d(
        teacher, kernel_size=3, stride=1, padding=1,
        count_include_pad=False)
    student_energy = student_high.square().sum(dim=1).sqrt().flatten(1)
    teacher_energy = teacher_high.square().sum(dim=1).sqrt().flatten(1)
    student_energy = F.normalize(student_energy, dim=1)
    teacher_energy = F.normalize(teacher_energy, dim=1)
    return 1.0 - (student_energy * teacher_energy).sum(dim=1)


def _projected_stats(raw: dict) -> dict[str, float | bool]:
    if not raw["finite_nonzero"]:
        return {
            "norm_ratio": float("nan"),
            "cosine": float("nan"),
            "retained_auxiliary_norm": float("nan"),
            "conflict_removed": False,
        }
    reference_sq = raw["reference_norm"] ** 2
    auxiliary_sq = raw["auxiliary_norm"] ** 2
    dot = raw["dot"]
    projected_sq = auxiliary_sq
    projected_dot = dot
    if dot < 0:
        projected_sq = max(0.0, auxiliary_sq - dot * dot / reference_sq)
        projected_dot = 0.0
    projected_norm = math.sqrt(projected_sq)
    denominator = raw["reference_norm"] * projected_norm
    return {
        "norm_ratio": projected_norm / raw["reference_norm"],
        "cosine": projected_dot / denominator if denominator > 0 else 0.0,
        "retained_auxiliary_norm": (
            projected_norm / raw["auxiliary_norm"]
            if raw["auxiliary_norm"] > 0 else float("nan")
        ),
        "conflict_removed": bool(dot < 0),
    }


def _summarize(rows: list[dict], protocol: dict) -> dict:
    batches = protocol["batches"]
    valid = [row for row in rows if row["gradient"]["finite_nonzero"]]
    raw_cosines = [row["gradient"]["cosine"] for row in valid]
    raw_ratios = [row["gradient"]["norm_ratio"] for row in valid]
    projected = [row["projected_gradient"] for row in valid]
    micro_gt = sum(row["micro_gt"] for row in rows)
    selected_gt = sum(row["selected_gt"] for row in rows)
    summary = {
        "batches": len(rows),
        "valid_gradient_batches": len(valid),
        "valid_batch_rate": len(valid) / len(rows) if rows else 0.0,
        "micro_gt": micro_gt,
        "selected_gt": selected_gt,
        "selection_coverage": (
            selected_gt / micro_gt if micro_gt else float("nan")),
        "weight_sum": sum(row["weight_sum"] for row in rows),
        "raw_conflict_batches": sum(
            bool(row["gradient"]["conflict"]) for row in valid),
        "raw_conflict_rate": (
            sum(bool(row["gradient"]["conflict"]) for row in valid)
            / len(valid) if valid else float("nan")),
        "raw_cosine_mean": (
            statistics.fmean(raw_cosines) if raw_cosines else float("nan")),
        "raw_cosine_median": (
            statistics.median(raw_cosines) if raw_cosines else float("nan")),
        "raw_cosine_q1": _quantile(raw_cosines, 0.25),
        "raw_cosine_q3": _quantile(raw_cosines, 0.75),
        "raw_norm_ratio_mean": (
            statistics.fmean(raw_ratios) if raw_ratios else float("nan")),
        "raw_norm_ratio_median": (
            statistics.median(raw_ratios) if raw_ratios else float("nan")),
        "projected_cosine_mean": (
            statistics.fmean(item["cosine"] for item in projected)
            if projected else float("nan")),
        "projected_norm_ratio_mean": (
            statistics.fmean(item["norm_ratio"] for item in projected)
            if projected else float("nan")),
        "projected_retained_norm_mean": (
            statistics.fmean(
                item["retained_auxiliary_norm"] for item in projected)
            if projected else float("nan")),
    }
    summary["probe_gate"] = {
        "complete": len(rows) == batches,
        "valid_batch_rate_min": 0.50,
        "selection_coverage_min": 0.10,
        "raw_cosine_mean_min": -0.15,
        "projected_retained_norm_mean_min": 0.50,
    }
    summary["probe_pass"] = bool(
        len(rows) == batches
        and summary["valid_batch_rate"] >= 0.50
        and summary["selection_coverage"] >= 0.10
        and summary["raw_cosine_mean"] >= -0.15
        and summary["projected_retained_norm_mean"] >= 0.50
    )
    if protocol["full_gate"]:
        raw_conflict_rate_min = (
            0.35
            if protocol["feature_target_id"] == "high_frequency_energy"
            else 0.50
        )
        summary["full_gate"] = {
            "required_batches": 200,
            "valid_batch_rate_min": 0.60,
            "selection_coverage_min": 0.12,
            "raw_conflict_rate_min": raw_conflict_rate_min,
            "projected_cosine_mean_min": 0.0,
            "projected_norm_ratio_range": [0.03, 0.10],
            "projected_retained_norm_mean_min": 0.95,
        }
        summary["gate_pass"] = bool(
            len(rows) == batches == 200
            and summary["valid_batch_rate"] >= 0.60
            and summary["selection_coverage"] >= 0.12
            and summary["raw_conflict_rate"] >= raw_conflict_rate_min
            and summary["projected_cosine_mean"] >= 0.0
            and 0.03 <= summary["projected_norm_ratio_mean"] <= 0.10
            and summary["projected_retained_norm_mean"] >= 0.95
        )
    return summary


def _write_artifact(
    path: Path,
    protocol: dict,
    rows: list[dict],
    complete: bool,
    elapsed_seconds: float,
) -> None:
    artifact = {
        "complete": complete,
        "protocol": protocol,
        "elapsed_seconds": elapsed_seconds,
        "summary": _summarize(rows, protocol),
        "batches": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.batches <= 0 or args.batch_size <= 0 or args.save_every <= 0:
        raise ValueError("batches, batch-size, and save-every must be positive")
    if not 0 < args.loss_weight:
        raise ValueError("loss-weight must be positive")
    if args.spatial_temperature <= 0:
        raise ValueError("spatial-temperature must be positive")
    if args.full_gate and args.batches != 200:
        raise ValueError("The frozen full gate requires exactly 200 batches")
    if (
        args.full_gate
        and args.feature_target == "cosine"
        and not math.isclose(args.loss_weight, 0.15)
    ):
        raise ValueError("The frozen full gate requires loss-weight 0.15")
    if (
        args.full_gate
        and args.feature_target == "spatial_dependency"
        and (
            not math.isclose(args.loss_weight, 0.25)
            or not math.isclose(args.spatial_temperature, 0.20)
        )
    ):
        raise ValueError(
            "The frozen PC-MSDD gate requires weight 0.25 and temperature 0.20"
        )
    if (
        args.full_gate
        and args.feature_target == "high_frequency"
        and not math.isclose(args.loss_weight, 0.20)
    ):
        raise ValueError(
            "The frozen PC-MHFD gate requires loss-weight 0.20"
        )
    if (
        args.full_gate
        and args.feature_target == "high_frequency_energy"
        and not math.isclose(args.loss_weight, 1.0)
    ):
        raise ValueError(
            "The frozen PC-MHED gate requires loss-weight 1.0"
        )
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
        pin_memory=(DEVICE.type == "cuda"),
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
        transform_min_sizes=(args.student_min_size,),
        transform_max_size=args.student_max_size,
    ).to(DEVICE)
    checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
    teacher = deepcopy(student)
    teacher.load_state_dict(checkpoint["model"])
    teacher.to(DEVICE).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    shared_parameters = tuple(
        parameter for parameter in student.backbone.fpn.parameters()
        if parameter.requires_grad
    )
    if not shared_parameters:
        raise RuntimeError("Student FPN has no trainable parameters")
    student.train()

    amp_enabled = DEVICE.type == "cuda" and not args.no_amp
    protocol = {
        "method": (
            "projected_micro_spatial_dependency_distillation_probe"
            if args.feature_target == "spatial_dependency" else
            "projected_micro_high_frequency_distillation_probe"
            if args.feature_target == "high_frequency" else
            "projected_micro_high_frequency_energy_distillation_probe"
            if args.feature_target == "high_frequency_energy" else
            "micro_object_centric_feature_distillation_probe"
        ),
        "seed": args.seed,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "weighted_sampler": True,
        "copy_paste": bool(copy_paste_pool),
        "student_initialization": "torchvision FasterRCNN ResNet50-FPN DEFAULT",
        "student_scale": [args.student_min_size, args.student_max_size],
        "teacher_path": str(teacher_path),
        "teacher_epoch": checkpoint.get("epoch"),
        "teacher_model_source": checkpoint.get("model_source"),
        "teacher_scale": [args.teacher_min_size, args.teacher_max_size],
        "proposal_top_n": args.proposal_top_n,
        "micro_cutoff_px": args.micro_cutoff_px,
        "teacher_iou_floor": args.teacher_iou_floor,
        "advantage_margin": args.advantage_margin,
        "loss_weight": args.loss_weight,
        "probe_scaling_basis": (
            "20-batch weight-0.20 probe projected norm 0.0080947; "
            "weight frozen at 1.0 before the 200-batch audit"
            if args.feature_target == "high_frequency_energy" else None
        ),
        "feature_target": (
            "exact-GT 7x7 FPN RoI, off-diagonal 49x49 spatial "
            "cosine-relation KL"
            if args.feature_target == "spatial_dependency" else
            "exact-GT 7x7 FPN RoI, teacher-energy-weighted 3x3 local "
            "residual cosine"
            if args.feature_target == "high_frequency" else
            "exact-GT 7x7 FPN RoI, channel-invariant normalized 3x3 local "
            "residual-energy-map cosine"
            if args.feature_target == "high_frequency_energy" else
            "exact-GT 7x7 FPN RoI, channel-normalized cosine"
        ),
        "feature_target_id": args.feature_target,
        "spatial_temperature": (
            args.spatial_temperature
            if args.feature_target == "spatial_dependency" else None
        ),
        "spatial_loss_scaling": (
            "mean KL without temperature-squared rescaling"
            if args.feature_target == "spatial_dependency" else None
        ),
        "auxiliary_parameters": "student FPN only; body detached",
        "detector_loss": "sum(all standard detector losses)",
        "amp": amp_enabled,
        "optimizer_updates": 0,
        "claim_boundary": (
            "Gate0 may authorize technical implementation only; no performance promotion"
            if args.full_gate else
            "mechanism probe only; no implementation/promotion"
        ),
        "full_gate": args.full_gate,
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
            detector_loss = sum(detector_losses.values())
        detector_gradients = torch.autograd.grad(
            detector_loss,
            shared_parameters,
            allow_unused=True,
        )

        student.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            student_snapshot = _fpn_snapshot(
                student,
                images,
                targets,
                args.student_min_size,
                args.student_max_size,
                require_fpn_grad=True,
            )
            teacher_snapshot = _fpn_snapshot(
                teacher,
                images,
                targets,
                args.teacher_min_size,
                args.teacher_max_size,
                require_fpn_grad=False,
            )
            feature_loss, telemetry = _feature_loss(
                student_snapshot,
                teacher_snapshot,
                targets,
                args.proposal_top_n,
                args.micro_cutoff_px,
                args.teacher_iou_floor,
                args.advantage_margin,
                args.loss_weight,
                args.feature_target,
                args.spatial_temperature,
            )
        feature_gradients = torch.autograd.grad(
            feature_loss,
            shared_parameters,
            allow_unused=True,
        )
        gradient = _gradient_pair_stats(
            detector_gradients, feature_gradients)
        row = {
            "batch": batch_index,
            "detector_loss": float(detector_loss.detach().item()),
            "feature_loss": float(feature_loss.detach().item()),
            **telemetry,
            "gradient": gradient,
            "projected_gradient": _projected_stats(gradient),
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
            summary = _summarize(rows, protocol)
            print(
                f"[{batch_index:03d}/{args.batches}] "
                f"coverage={summary['selection_coverage']:.3f}, "
                f"conflict={summary['raw_conflict_rate']:.3f}, "
                f"cos={summary['raw_cosine_mean']:.4f}, "
                f"ratio={summary['raw_norm_ratio_mean']:.4f}, "
                f"elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

    artifact = json.loads(args.out_json.read_text(encoding="utf-8"))
    artifact["peak_allocated_gib"] = (
        torch.cuda.max_memory_allocated() / 1024**3
        if DEVICE.type == "cuda" else 0.0
    )
    args.out_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))
    print(f"Saved audit: {args.out_json}")


if __name__ == "__main__":
    main()
