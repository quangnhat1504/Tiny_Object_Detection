"""Audit PC-MR-RPN and PC-MOC-FD gradient compatibility without updates."""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
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
from scripts.audit_cbl_cross_scale_gradient_conflict import (
    _gradient_pair_stats,
    _quantile,
)
from scripts.audit_ra_tb_pcmhfd_compatibility import (
    _add_gradients,
    _project_auxiliary,
)


DEFAULT_TEACHER = (
    ROOT
    / ".runtime/kaggle/cbl_iterative_train_fair20/output/tod_output/runs"
    / "sa_alw_full__cbl__irtw0.5ir1s0.3__la_loss__seed42__cbl_iterative_train_fair20"
    / "best.pt"
)
DEFAULT_OUTPUT = ROOT / "runs/pc_mr_moc_gradient_compatibility_seed42.json"
MR_LOSS_KEY = "loss_rpn_micro_rescue"
MOC_LOSS_KEY = "loss_fpn_micro_feature"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--no-amp", action="store_true")
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


def _summarize_metric(rows: list[dict], key: str) -> dict:
    valid = [row[key] for row in rows if row[key]["finite_nonzero"]]
    cosines = [float(item["cosine"]) for item in valid]
    ratios = [float(item["norm_ratio"]) for item in valid]
    return {
        "valid_batches": len(valid),
        "conflict_batches": sum(bool(item["conflict"]) for item in valid),
        "conflict_rate": (
            sum(bool(item["conflict"]) for item in valid) / len(valid)
            if valid else float("nan")
        ),
        "cosine_mean": statistics.fmean(cosines) if cosines else float("nan"),
        "cosine_median": statistics.median(cosines) if cosines else float("nan"),
        "cosine_q1": _quantile(cosines, 0.25),
        "cosine_q3": _quantile(cosines, 0.75),
        "norm_ratio_mean": statistics.fmean(ratios) if ratios else float("nan"),
        "norm_ratio_median": statistics.median(ratios) if ratios else float("nan"),
    }


def _cross_scope_stats(reference, auxiliary) -> dict[str, float | bool | int]:
    """Measure two gradient vectors even when their parameter support is disjoint."""
    dot = 0.0
    reference_sq = 0.0
    auxiliary_sq = 0.0
    overlapping_positions = 0
    for reference_gradient, auxiliary_gradient in zip(reference, auxiliary):
        if reference_gradient is not None:
            reference_sq += float(
                reference_gradient.detach().float().square().sum().item())
        if auxiliary_gradient is not None:
            auxiliary_sq += float(
                auxiliary_gradient.detach().float().square().sum().item())
        if reference_gradient is not None and auxiliary_gradient is not None:
            overlapping_positions += 1
            dot += float((
                reference_gradient.detach().float()
                * auxiliary_gradient.detach().float()
            ).sum().item())
    reference_norm = math.sqrt(reference_sq)
    auxiliary_norm = math.sqrt(auxiliary_sq)
    finite_nonzero = bool(
        math.isfinite(reference_norm)
        and math.isfinite(auxiliary_norm)
        and reference_norm > 0
        and auxiliary_norm > 0
    )
    cosine = (
        dot / (reference_norm * auxiliary_norm)
        if finite_nonzero else float("nan")
    )
    return {
        "dot": dot,
        "reference_norm": reference_norm,
        "auxiliary_norm": auxiliary_norm,
        "norm_ratio": (
            auxiliary_norm / reference_norm
            if finite_nonzero else float("nan")
        ),
        "cosine": cosine,
        "conflict": bool(finite_nonzero and cosine < 0),
        "finite_nonzero": finite_nonzero,
        "overlapping_parameter_positions": overlapping_positions,
        "disjoint_support": overlapping_positions == 0,
    }


def _summarize(rows: list[dict], requested_batches: int) -> dict:
    keys = (
        "detector_match",
        "detector_pc_mr_raw",
        "detector_pc_mr_projected",
        "detector_pc_moc_raw",
        "detector_pc_moc_projected",
        "auxiliary_cross_scope",
        "detector_final_update",
    )
    summary = {key: _summarize_metric(rows, key) for key in keys}
    jointly_valid = [
        row for row in rows
        if row["detector_pc_mr_raw"]["finite_nonzero"]
        and row["detector_pc_moc_raw"]["finite_nonzero"]
    ]
    summary.update({
        "batches": len(rows),
        "joint_valid_batches": len(jointly_valid),
        "joint_valid_rate": len(jointly_valid) / len(rows) if rows else 0.0,
        "selection_identity_rate": (
            sum(row["selection_identity"] for row in rows) / len(rows)
            if rows else 0.0
        ),
        "disjoint_support_rate": (
            sum(
                row["auxiliary_cross_scope"]["disjoint_support"]
                for row in rows
            ) / len(rows) if rows else 0.0
        ),
        "pc_mr_selected_gt": sum(row["pc_mr_selected_gt"] for row in rows),
        "pc_moc_selected_gt": sum(row["pc_moc_selected_gt"] for row in rows),
        "micro_gt": sum(row["micro_gt"] for row in rows),
        "pc_mr_projected_retained_norm_mean": (
            statistics.fmean(
                row["pc_mr_projection"]["retained_auxiliary_norm"]
                for row in jointly_valid
            ) if jointly_valid else float("nan")
        ),
        "pc_moc_projected_retained_norm_mean": (
            statistics.fmean(
                row["pc_moc_projection"]["retained_auxiliary_norm"]
                for row in jointly_valid
            ) if jointly_valid else float("nan")
        ),
    })
    summary["probe_gate"] = {
        "minimum_batches": 20,
        "joint_valid_rate_min": 0.60,
        "detector_match_cosine_min": 0.9999,
        "selection_identity_rate_min": 0.99,
        "projected_cosine_mean_min": 0.0,
        "projected_retained_norm_mean_min": 0.95,
        "auxiliary_cross_scope_abs_cosine_max": 1e-7,
        "detector_final_update_cosine_mean_min": 0.95,
        "detector_final_update_norm_ratio_range": [0.90, 1.20],
    }
    detector_match = summary["detector_match"]
    pc_mr_projected = summary["detector_pc_mr_projected"]
    pc_moc_projected = summary["detector_pc_moc_projected"]
    cross_scope = summary["auxiliary_cross_scope"]
    final_update = summary["detector_final_update"]
    summary["probe_pass"] = bool(
        len(rows) >= 20
        and summary["joint_valid_rate"] >= 0.60
        and detector_match["valid_batches"] == len(rows)
        and detector_match["cosine_mean"] >= 0.9999
        and 0.999 <= detector_match["norm_ratio_mean"] <= 1.001
        and summary["selection_identity_rate"] >= 0.99
        and summary["disjoint_support_rate"] == 1.0
        and pc_mr_projected["cosine_mean"] >= 0.0
        and pc_moc_projected["cosine_mean"] >= 0.0
        and summary["pc_mr_projected_retained_norm_mean"] >= 0.95
        and summary["pc_moc_projected_retained_norm_mean"] >= 0.95
        and abs(cross_scope["cosine_mean"]) <= 1e-7
        and final_update["cosine_mean"] >= 0.95
        and 0.90 <= final_update["norm_ratio_mean"] <= 1.20
    )
    summary["full_gate"] = {
        "required_batches": 200,
        "same_thresholds_as_probe": True,
    }
    summary["gate_pass"] = bool(
        requested_batches == 200
        and len(rows) == 200
        and summary["probe_pass"]
    )
    return summary


def _write(path: Path, protocol: dict, rows: list[dict], elapsed: float) -> None:
    artifact = {
        "complete": len(rows) == protocol["batches"],
        "protocol": protocol,
        "elapsed_seconds": elapsed,
        "summary": _summarize(rows, protocol["batches"]),
        "batches": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if DEVICE.type != "cuda":
        raise RuntimeError("PC-MR + PC-MOC compatibility audit requires CUDA")
    if args.batches <= 0 or args.batch_size <= 0 or args.save_every <= 0:
        raise ValueError("batches, batch-size, and save-every must be positive")
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
        num_samples=args.batches * args.batch_size,
        replacement=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )

    pc_mr_student = _build(reliability_threshold)
    pc_moc_student = deepcopy(pc_mr_student)
    teacher_checkpoint = torch.load(
        teacher_path, map_location="cpu", weights_only=False)
    teacher = deepcopy(pc_mr_student)
    teacher.load_state_dict(teacher_checkpoint["model"])
    teacher.to(DEVICE).eval()
    attach_pc_micro_rescue_rpn_teacher(
        pc_mr_student,
        teacher,
        loss_weight=0.005,
        teacher_min_size=960,
        teacher_max_size=1200,
        proposal_top_n=300,
        micro_cutoff_px=8.0,
        teacher_iou_floor=0.50,
        advantage_margin=0.02,
    )
    attach_pc_micro_object_feature_teacher(
        pc_moc_student,
        teacher,
        loss_weight=0.15,
        teacher_min_size=960,
        teacher_max_size=1200,
        proposal_top_n=300,
        micro_cutoff_px=8.0,
        teacher_iou_floor=0.50,
        advantage_margin=0.02,
        feature_target="cosine",
    )
    pc_mr_student.train()
    pc_moc_student.train()
    teacher.eval()

    pc_mr_fpn_parameters = tuple(
        parameter for parameter in pc_mr_student.backbone.fpn.parameters()
        if parameter.requires_grad)
    pc_mr_rpn_parameters = tuple(
        parameter for parameter in pc_mr_student.rpn.head.parameters()
        if parameter.requires_grad)
    pc_moc_fpn_parameters = tuple(
        parameter for parameter in pc_moc_student.backbone.fpn.parameters()
        if parameter.requires_grad)
    pc_moc_rpn_parameters = tuple(
        parameter for parameter in pc_moc_student.rpn.head.parameters()
        if parameter.requires_grad)
    if len(pc_mr_fpn_parameters) != len(pc_moc_fpn_parameters):
        raise AssertionError("Student FPN parameter topology differs")
    if len(pc_mr_rpn_parameters) != len(pc_moc_rpn_parameters):
        raise AssertionError("Student RPN parameter topology differs")
    if set(map(id, pc_mr_fpn_parameters)) & set(map(id, pc_mr_rpn_parameters)):
        raise AssertionError("FPN and RPN PCGrad parameter scopes overlap")
    if getattr(pc_mr_student.rpn, "_micro_rescue_teacher") is not teacher:
        raise AssertionError("PC-MR did not retain the shared teacher")
    if getattr(pc_moc_student.backbone, "_moc_feature_teacher") is not teacher:
        raise AssertionError("PC-MOC did not retain the shared teacher")
    if any("teacher" in key for key in pc_mr_student.state_dict()):
        raise AssertionError("PC-MR teacher leaked into student state_dict")
    if any("teacher" in key for key in pc_moc_student.state_dict()):
        raise AssertionError("PC-MOC teacher leaked into student state_dict")

    amp_enabled = not args.no_amp
    protocol = {
        "method": "pc_mr_rpn_plus_pc_moc_fd_gradient_compatibility",
        "seed": args.seed,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "weighted_sampler": True,
        "copy_paste": bool(copy_paste_pool),
        "student_initialization": "identical torchvision FasterRCNN ResNet50-FPN DEFAULT",
        "student_scale": [640, 800],
        "teacher_path": str(teacher_path),
        "teacher_epoch": teacher_checkpoint.get("epoch"),
        "teacher_model_source": teacher_checkpoint.get("model_source"),
        "teacher_scale": [960, 1200],
        "shared_teacher_object": True,
        "pc_mr_rpn": {
            "weight": 0.005,
            "pcgrad_scope": "student RPN head",
        },
        "pc_moc_fd": {
            "weight": 0.15,
            "feature_target": "cosine",
            "pcgrad_scope": "student FPN",
        },
        "proposal_top_n": 300,
        "micro_cutoff_px": 8.0,
        "teacher_iou_floor": 0.50,
        "advantage_margin": 0.02,
        "optimizer_updates": 0,
        "amp": amp_enabled,
        "claim_boundary": (
            "Compatibility gate only; it does not authorize performance or locked-test claims"
        ),
    }
    del teacher_checkpoint

    rows = []
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images, targets = _move_batch(images, targets)
        batch_seed = args.seed * 1000 + batch_index

        seed_all(batch_seed)
        pc_mr_student.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            pc_mr_losses = pc_mr_student(images, targets)
            pc_mr_auxiliary = pc_mr_losses[MR_LOSS_KEY]
            pc_mr_detector = sum(
                value for name, value in pc_mr_losses.items()
                if name != MR_LOSS_KEY)
        pc_mr_stats = dict(pc_mr_student.rpn._micro_rescue_stats)
        pc_mr_union_parameters = pc_mr_fpn_parameters + pc_mr_rpn_parameters
        pc_mr_detector_gradients = torch.autograd.grad(
            pc_mr_detector,
            pc_mr_union_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        pc_mr_auxiliary_gradients = torch.autograd.grad(
            pc_mr_auxiliary,
            pc_mr_union_parameters,
            allow_unused=True,
        )

        seed_all(batch_seed)
        pc_moc_student.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            pc_moc_losses = pc_moc_student(images, targets)
            pc_moc_auxiliary = pc_moc_losses[MOC_LOSS_KEY]
            pc_moc_detector = sum(
                value for name, value in pc_moc_losses.items()
                if name != MOC_LOSS_KEY)
        pc_moc_stats = dict(pc_moc_student.backbone._moc_feature_stats)
        pc_moc_union_parameters = pc_moc_fpn_parameters + pc_moc_rpn_parameters
        pc_moc_detector_gradients = torch.autograd.grad(
            pc_moc_detector,
            pc_moc_union_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        pc_moc_auxiliary_gradients = torch.autograd.grad(
            pc_moc_auxiliary,
            pc_moc_union_parameters,
            allow_unused=True,
        )

        fpn_count = len(pc_mr_fpn_parameters)
        detector_fpn = pc_mr_detector_gradients[:fpn_count]
        detector_rpn = pc_mr_detector_gradients[fpn_count:]
        pc_mr_auxiliary_rpn = pc_mr_auxiliary_gradients[fpn_count:]
        pc_moc_auxiliary_fpn = pc_moc_auxiliary_gradients[:fpn_count]
        projected_pc_mr, pc_mr_projection = _project_auxiliary(
            detector_rpn, pc_mr_auxiliary_rpn)
        projected_pc_moc, pc_moc_projection = _project_auxiliary(
            detector_fpn, pc_moc_auxiliary_fpn)
        final_update = _add_gradients(
            pc_mr_detector_gradients,
            projected_pc_moc + (None,) * len(pc_mr_rpn_parameters),
            (None,) * fpn_count + projected_pc_mr,
        )
        pc_mr_auxiliary_union = (
            (None,) * fpn_count + pc_mr_auxiliary_rpn)
        pc_moc_auxiliary_union = (
            pc_moc_auxiliary_fpn + (None,) * len(pc_mr_rpn_parameters))

        selection_identity = bool(
            int(pc_mr_stats.get("micro_gt", 0))
            == int(pc_moc_stats.get("micro_gt", 0))
            and int(pc_mr_stats.get("selected_gt", 0))
            == int(pc_moc_stats.get("selected_gt", 0))
            and abs(
                float(pc_mr_stats.get("teacher_iou_mean", 0.0))
                - float(pc_moc_stats.get("teacher_iou_mean", 0.0))
            ) <= 1e-6
            and abs(
                float(pc_mr_stats.get("student_iou_mean", 0.0))
                - float(pc_moc_stats.get("student_iou_mean", 0.0))
            ) <= 1e-6
        )
        row = {
            "batch": batch_index,
            "pc_mr_detector_loss": float(pc_mr_detector.detach()),
            "pc_moc_detector_loss": float(pc_moc_detector.detach()),
            "pc_mr_auxiliary_loss": float(pc_mr_auxiliary.detach()),
            "pc_moc_auxiliary_loss": float(pc_moc_auxiliary.detach()),
            "micro_gt": int(pc_mr_stats.get("micro_gt", 0)),
            "pc_mr_selected_gt": int(pc_mr_stats.get("selected_gt", 0)),
            "pc_moc_selected_gt": int(pc_moc_stats.get("selected_gt", 0)),
            "selection_identity": selection_identity,
            "detector_match": _gradient_pair_stats(
                pc_mr_detector_gradients, pc_moc_detector_gradients),
            "detector_pc_mr_raw": _gradient_pair_stats(
                detector_rpn, pc_mr_auxiliary_rpn),
            "detector_pc_mr_projected": _gradient_pair_stats(
                detector_rpn, projected_pc_mr),
            "detector_pc_moc_raw": _gradient_pair_stats(
                detector_fpn, pc_moc_auxiliary_fpn),
            "detector_pc_moc_projected": _gradient_pair_stats(
                detector_fpn, projected_pc_moc),
            "auxiliary_cross_scope": _cross_scope_stats(
                pc_mr_auxiliary_union, pc_moc_auxiliary_union),
            "detector_final_update": _gradient_pair_stats(
                pc_mr_detector_gradients, final_update),
            "pc_mr_projection": pc_mr_projection,
            "pc_moc_projection": pc_moc_projection,
        }
        rows.append(row)
        if not selection_identity:
            raise AssertionError(
                f"PC-MR and PC-MOC selected different GTs at batch {batch_index}")
        if any(parameter.grad is not None for parameter in teacher.parameters()):
            raise AssertionError("Frozen shared teacher received gradients")

        del (
            pc_mr_losses,
            pc_mr_detector,
            pc_mr_auxiliary,
            pc_mr_detector_gradients,
            pc_mr_auxiliary_gradients,
            pc_mr_stats,
            pc_moc_losses,
            pc_moc_detector,
            pc_moc_auxiliary,
            pc_moc_detector_gradients,
            pc_moc_auxiliary_gradients,
            pc_moc_stats,
            detector_fpn,
            detector_rpn,
            pc_mr_auxiliary_rpn,
            pc_moc_auxiliary_fpn,
            projected_pc_mr,
            projected_pc_moc,
            final_update,
            pc_mr_auxiliary_union,
            pc_moc_auxiliary_union,
        )
        if batch_index % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

        if batch_index % args.save_every == 0 or batch_index == args.batches:
            elapsed = time.time() - started
            _write(args.out_json, protocol, rows, elapsed)
            summary = _summarize(rows, args.batches)
            print(
                f"[{batch_index:03d}/{args.batches}] "
                f"joint={summary['joint_valid_rate']:.3f}, "
                f"mr={summary['detector_pc_mr_projected']['cosine_mean']:.4f}, "
                f"moc={summary['detector_pc_moc_projected']['cosine_mean']:.4f}, "
                f"final={summary['detector_final_update']['cosine_mean']:.4f}, "
                f"ratio={summary['detector_final_update']['norm_ratio_mean']:.4f}, "
                f"elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

    artifact = json.loads(args.out_json.read_text(encoding="utf-8"))
    artifact["peak_allocated_gib"] = (
        torch.cuda.max_memory_allocated() / 1024**3)
    args.out_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))
    print(f"Saved audit: {args.out_json}")


if __name__ == "__main__":
    main()
