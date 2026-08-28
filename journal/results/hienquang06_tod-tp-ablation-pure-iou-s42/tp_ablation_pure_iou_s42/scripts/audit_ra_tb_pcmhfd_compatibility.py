"""Audit RA-TB and PC-MHFD gradient compatibility without optimizer updates."""

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
    attach_cbl_cross_scale_teacher,
    attach_pc_micro_object_feature_teacher,
    build_model,
)
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
DEFAULT_OUTPUT = ROOT / "runs/ra_tb_pcmhfd_fpn_compatibility_seed42.json"
RA_LOSS_KEY = "loss_box_scale_distill"
MHFD_LOSS_KEY = "loss_fpn_micro_feature"


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


def _add_gradients(*gradient_groups):
    combined = []
    for gradients in zip(*gradient_groups):
        tensors = [gradient for gradient in gradients if gradient is not None]
        combined.append(sum(tensors) if tensors else None)
    return tuple(combined)


def _project_auxiliary(reference, auxiliary):
    stats = _gradient_pair_stats(reference, auxiliary)
    if not stats["finite_nonzero"] or stats["dot"] >= 0:
        return tuple(
            gradient.detach() if gradient is not None else None
            for gradient in auxiliary
        ), {
            "applied": False,
            "retained_auxiliary_norm": 1.0,
        }

    reference_sq = stats["reference_norm"] ** 2
    coefficient = stats["dot"] / reference_sq
    projected = []
    projected_sq = 0.0
    for reference_gradient, auxiliary_gradient in zip(reference, auxiliary):
        if auxiliary_gradient is None:
            projected.append(None)
            continue
        candidate = auxiliary_gradient.detach()
        if reference_gradient is not None:
            candidate = candidate - coefficient * reference_gradient.detach()
        projected.append(candidate)
        projected_sq += float(candidate.float().square().sum().item())
    projected_norm = math.sqrt(projected_sq)
    retained = projected_norm / stats["auxiliary_norm"]
    return tuple(projected), {
        "applied": True,
        "retained_auxiliary_norm": retained,
    }


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


def _summarize(rows: list[dict], batches: int) -> dict:
    keys = (
        "detector_match",
        "detector_ra",
        "detector_mhfd",
        "ra_mhfd",
        "detector_ra_reference",
        "reference_mhfd_raw",
        "reference_mhfd_projected",
        "detector_final_update",
    )
    summary = {key: _summarize_metric(rows, key) for key in keys}
    jointly_valid = [
        row for row in rows
        if row["detector_ra"]["finite_nonzero"]
        and row["detector_mhfd"]["finite_nonzero"]
    ]
    summary.update({
        "batches": len(rows),
        "joint_valid_batches": len(jointly_valid),
        "joint_valid_rate": len(jointly_valid) / len(rows) if rows else 0.0,
        "mhfd_projection_rate": (
            sum(row["projection"]["applied"] for row in jointly_valid)
            / len(jointly_valid) if jointly_valid else float("nan")
        ),
        "projected_retained_norm_mean": (
            statistics.fmean(
                row["projection"]["retained_auxiliary_norm"]
                for row in jointly_valid
            ) if jointly_valid else float("nan")
        ),
        "ra_selected_coordinates": sum(
            row["ra_selected_coordinates"] for row in rows),
        "ra_positive_coordinates": sum(
            row["ra_positive_coordinates"] for row in rows),
        "mhfd_selected_gt": sum(row["mhfd_selected_gt"] for row in rows),
        "mhfd_micro_gt": sum(row["mhfd_micro_gt"] for row in rows),
    })
    summary["gate"] = {
        "required_batches": 200,
        "joint_valid_rate_min": 0.60,
        "detector_match_cosine_min": 0.9999,
        "detector_match_norm_ratio_range": [0.999, 1.001],
        "ra_mhfd_cosine_mean_min": -0.10,
        "detector_ra_reference_cosine_mean_min": 0.95,
        "reference_mhfd_projected_cosine_mean_min": 0.0,
        "projected_retained_norm_mean_min": 0.95,
        "detector_final_update_cosine_mean_min": 0.95,
        "detector_final_update_norm_ratio_range": [0.90, 1.20],
    }
    detector_match = summary["detector_match"]
    final_update = summary["detector_final_update"]
    summary["gate_pass"] = bool(
        len(rows) == batches == 200
        and summary["joint_valid_rate"] >= 0.60
        and detector_match["valid_batches"] == 200
        and detector_match["cosine_mean"] >= 0.9999
        and 0.999 <= detector_match["norm_ratio_mean"] <= 1.001
        and summary["ra_mhfd"]["cosine_mean"] >= -0.10
        and summary["detector_ra_reference"]["cosine_mean"] >= 0.95
        and summary["reference_mhfd_projected"]["cosine_mean"] >= 0.0
        and summary["projected_retained_norm_mean"] >= 0.95
        and final_update["cosine_mean"] >= 0.95
        and 0.90 <= final_update["norm_ratio_mean"] <= 1.20
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
        raise RuntimeError("RA-TB + PC-MHFD compatibility audit requires CUDA")
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

    ra_student = _build(reliability_threshold)
    mhfd_student = deepcopy(ra_student)
    teacher_checkpoint = torch.load(
        teacher_path, map_location="cpu", weights_only=False)
    teacher = deepcopy(ra_student)
    teacher.load_state_dict(teacher_checkpoint["model"])
    teacher.to(DEVICE).eval()
    attach_cbl_cross_scale_teacher(
        ra_student,
        teacher,
        loss_weight=0.25,
        temperature=2.0,
        advantage_margin=0.02,
        teacher_min_size=960,
        teacher_max_size=1200,
        tiny_reference=16.0,
        tiny_weight_cap=2.0,
        coordinate_reliable=True,
        distill_distance="teacher_bounded_gt",
        distill_stage="refined",
    )
    attach_pc_micro_object_feature_teacher(
        mhfd_student,
        teacher,
        loss_weight=0.20,
        teacher_min_size=960,
        teacher_max_size=1200,
        proposal_top_n=300,
        micro_cutoff_px=8.0,
        teacher_iou_floor=0.50,
        advantage_margin=0.02,
        feature_target="high_frequency",
    )
    ra_student.roi_heads._cbl_scale_collect_audit = True
    ra_student.train()
    mhfd_student.train()
    teacher.eval()

    ra_parameters = tuple(
        parameter for parameter in ra_student.backbone.fpn.parameters()
        if parameter.requires_grad)
    mhfd_parameters = tuple(
        parameter for parameter in mhfd_student.backbone.fpn.parameters()
        if parameter.requires_grad)
    if len(ra_parameters) != len(mhfd_parameters):
        raise AssertionError("Student FPN parameter topology differs")
    if any("teacher" in key for key in ra_student.state_dict()):
        raise AssertionError("RA-TB teacher leaked into the student state dict")
    if any("teacher" in key for key in mhfd_student.state_dict()):
        raise AssertionError("PC-MHFD teacher leaked into the student state dict")

    amp_enabled = not args.no_amp
    protocol = {
        "method": "ra_tb_pc_mhfd_fpn_gradient_compatibility",
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
        "ra_tb": {
            "weight": 0.25,
            "stage": "refined",
            "distance": "teacher_bounded_gt",
            "coordinate_reliable": True,
        },
        "pc_mhfd": {
            "weight": 0.20,
            "feature_target": "high_frequency",
            "pcgrad_reference": "detector_plus_ra_tb_on_student_fpn",
        },
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
        ra_student.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            ra_losses = ra_student(images, targets)
            ra_auxiliary = ra_losses[RA_LOSS_KEY]
            ra_detector = sum(
                value for name, value in ra_losses.items()
                if name != RA_LOSS_KEY)
        ra_audit = ra_student.roi_heads._cbl_scale_audit
        if ra_audit is None:
            raise RuntimeError("RA-TB audit telemetry was not collected")
        ra_detector_gradients = torch.autograd.grad(
            ra_detector, ra_parameters, retain_graph=True, allow_unused=True)
        ra_auxiliary_gradients = torch.autograd.grad(
            ra_auxiliary, ra_parameters, allow_unused=True)

        seed_all(batch_seed)
        mhfd_student.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            mhfd_losses = mhfd_student(images, targets)
            mhfd_auxiliary = mhfd_losses[MHFD_LOSS_KEY]
            mhfd_detector = sum(
                value for name, value in mhfd_losses.items()
                if name != MHFD_LOSS_KEY)
        mhfd_stats = dict(mhfd_student.backbone._moc_feature_stats)
        mhfd_detector_gradients = torch.autograd.grad(
            mhfd_detector,
            mhfd_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        mhfd_auxiliary_gradients = torch.autograd.grad(
            mhfd_auxiliary, mhfd_parameters, allow_unused=True)

        reference_gradients = _add_gradients(
            ra_detector_gradients, ra_auxiliary_gradients)
        projected_mhfd, projection = _project_auxiliary(
            reference_gradients, mhfd_auxiliary_gradients)
        final_gradients = _add_gradients(reference_gradients, projected_mhfd)
        row = {
            "batch": batch_index,
            "ra_detector_loss": float(ra_detector.detach()),
            "mhfd_detector_loss": float(mhfd_detector.detach()),
            "ra_auxiliary_loss": float(ra_auxiliary.detach()),
            "mhfd_auxiliary_loss": float(mhfd_auxiliary.detach()),
            "ra_selected_coordinates": int(ra_audit["selected_coordinates"]),
            "ra_positive_coordinates": int(ra_audit["positive_coordinates"]),
            "mhfd_selected_gt": int(mhfd_stats.get("selected_gt", 0)),
            "mhfd_micro_gt": int(mhfd_stats.get("micro_gt", 0)),
            "detector_match": _gradient_pair_stats(
                ra_detector_gradients, mhfd_detector_gradients),
            "detector_ra": _gradient_pair_stats(
                ra_detector_gradients, ra_auxiliary_gradients),
            "detector_mhfd": _gradient_pair_stats(
                ra_detector_gradients, mhfd_auxiliary_gradients),
            "ra_mhfd": _gradient_pair_stats(
                ra_auxiliary_gradients, mhfd_auxiliary_gradients),
            "detector_ra_reference": _gradient_pair_stats(
                ra_detector_gradients, reference_gradients),
            "reference_mhfd_raw": _gradient_pair_stats(
                reference_gradients, mhfd_auxiliary_gradients),
            "reference_mhfd_projected": _gradient_pair_stats(
                reference_gradients, projected_mhfd),
            "detector_final_update": _gradient_pair_stats(
                ra_detector_gradients, final_gradients),
            "projection": projection,
        }
        rows.append(row)
        if any(parameter.grad is not None for parameter in teacher.parameters()):
            raise AssertionError("Frozen teacher received gradients")
        ra_student.roi_heads._cbl_scale_audit = None
        del (
            ra_losses,
            ra_detector,
            ra_auxiliary,
            ra_detector_gradients,
            ra_auxiliary_gradients,
            ra_audit,
            mhfd_losses,
            mhfd_detector,
            mhfd_auxiliary,
            mhfd_detector_gradients,
            mhfd_auxiliary_gradients,
            mhfd_stats,
            reference_gradients,
            projected_mhfd,
            final_gradients,
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
                f"ra_mhfd={summary['ra_mhfd']['cosine_mean']:.4f}, "
                f"final_det={summary['detector_final_update']['cosine_mean']:.4f}, "
                f"final_ratio={summary['detector_final_update']['norm_ratio_mean']:.4f}, "
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
