"""Audit detector/distillation gradient conflict without updating the model."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from copy import deepcopy
from itertools import chain
from pathlib import Path
from typing import Iterable

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
from common.model import build_model, attach_cbl_cross_scale_teacher


DEFAULT_TEACHER = (
    ROOT
    / ".runtime/kaggle/cbl_iterative_train_fair20/output/tod_output/runs"
    / "sa_alw_full__cbl__irtw0.5ir1s0.3__la_loss__seed42__cbl_iterative_train_fair20"
    / "best.pt"
)
DEFAULT_OUTPUT = ROOT / "runs/ca_sc_cbl_gradient_conflict_audit_seed42.json"
DEFAULT_CR_REFERENCE = ROOT / "runs/cr_sc_cbl_train_viability_audit_seed42.json"
DISTILL_LOSS_KEY = "loss_box_scale_distill"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure detector versus cross-scale CBL gradients on the shared "
            "RoI box head; no optimizer step is performed."
        )
    )
    parser.add_argument("--teacher", type=Path, default=DEFAULT_TEACHER)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--band-every", type=int, default=5)
    parser.add_argument("--coordinate-reliable", action="store_true")
    parser.add_argument(
        "--distill-distance",
        choices=("kl", "ordered_w1", "teacher_bounded_gt"),
        default="kl",
    )
    parser.add_argument(
        "--reference-json",
        type=Path,
        default=DEFAULT_CR_REFERENCE,
    )
    parser.add_argument("--cross-head", action="store_true")
    parser.add_argument(
        "--distill-stage", choices=("first", "refined"), default="first")
    parser.add_argument(
        "--shared-parameters",
        choices=("box_head", "representation"),
        default="box_head",
    )
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _gradient_pair_stats(
    reference: Iterable[torch.Tensor | None],
    auxiliary: Iterable[torch.Tensor | None],
) -> dict[str, float | bool]:
    dot = torch.zeros((), dtype=torch.float64)
    reference_sq = torch.zeros((), dtype=torch.float64)
    auxiliary_sq = torch.zeros((), dtype=torch.float64)
    for reference_grad, auxiliary_grad in zip(reference, auxiliary):
        if reference_grad is None or auxiliary_grad is None:
            continue
        reference_flat = reference_grad.detach().reshape(-1).float()
        auxiliary_flat = auxiliary_grad.detach().reshape(-1).float()
        dot += torch.dot(reference_flat, auxiliary_flat).double().cpu()
        reference_sq += torch.dot(
            reference_flat, reference_flat).double().cpu()
        auxiliary_sq += torch.dot(
            auxiliary_flat, auxiliary_flat).double().cpu()

    reference_norm = math.sqrt(float(reference_sq))
    auxiliary_norm = math.sqrt(float(auxiliary_sq))
    finite_nonzero = (
        math.isfinite(reference_norm)
        and math.isfinite(auxiliary_norm)
        and reference_norm > 0
        and auxiliary_norm > 0
    )
    cosine = (
        float(dot) / (reference_norm * auxiliary_norm)
        if finite_nonzero
        else float("nan")
    )
    return {
        "dot": float(dot),
        "reference_norm": reference_norm,
        "auxiliary_norm": auxiliary_norm,
        "norm_ratio": (
            auxiliary_norm / reference_norm
            if finite_nonzero
            else float("nan")
        ),
        "cosine": cosine,
        "conflict": bool(finite_nonzero and cosine < 0),
        "finite_nonzero": finite_nonzero,
    }


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _not_audited_stats() -> dict[str, float | bool]:
    return {
        "dot": float("nan"),
        "reference_norm": float("nan"),
        "auxiliary_norm": float("nan"),
        "norm_ratio": float("nan"),
        "cosine": float("nan"),
        "conflict": False,
        "finite_nonzero": False,
    }


def _summarize(rows: list[dict], key: str) -> dict[str, float | int]:
    valid = [row[key] for row in rows if row[key]["finite_nonzero"]]
    cosines = [float(item["cosine"]) for item in valid]
    ratios = [float(item["norm_ratio"]) for item in valid]
    conflicts = sum(bool(item["conflict"]) for item in valid)
    return {
        "valid_batches": len(valid),
        "conflict_batches": conflicts,
        "conflict_rate": conflicts / len(valid) if valid else float("nan"),
        "cosine_mean": statistics.fmean(cosines) if cosines else float("nan"),
        "cosine_median": statistics.median(cosines) if cosines else float("nan"),
        "cosine_q1": _quantile(cosines, 0.25),
        "cosine_q3": _quantile(cosines, 0.75),
        "norm_ratio_mean": statistics.fmean(ratios) if ratios else float("nan"),
        "norm_ratio_median": statistics.median(ratios) if ratios else float("nan"),
    }


def _write_artifact(
    path: Path,
    protocol: dict,
    rows: list[dict],
    complete: bool,
    elapsed_seconds: float,
) -> None:
    summary = {
        "overall": _summarize(rows, "overall"),
        "tiny": _summarize(rows, "tiny"),
        "larger": _summarize(rows, "larger"),
        "positive_rois": sum(row["positive_rois"] for row in rows),
        "selected_rois": sum(row["selected_rois"] for row in rows),
        "positive_coordinates": sum(
            row["positive_coordinates"] for row in rows),
        "selected_coordinates": sum(
            row["selected_coordinates"] for row in rows),
        "coordinate_weight_sum": sum(
            row["coordinate_weight_sum"] for row in rows),
        "selected_tiny_rois": sum(row["selected_tiny_rois"] for row in rows),
        "selected_larger_rois": sum(
            row["selected_larger_rois"] for row in rows),
        "positive_tiny_rois": sum(
            row["positive_tiny_rois"] for row in rows),
        "positive_larger_rois": sum(
            row["positive_larger_rois"] for row in rows),
    }
    overall = summary["overall"]
    if protocol["coordinate_reliable"]:
        positive_coordinates = summary["positive_coordinates"]
        selected_coordinates = summary["selected_coordinates"]
        summary["selected_coordinate_coverage"] = (
            selected_coordinates / positive_coordinates
            if positive_coordinates
            else float("nan")
        )
        summary["mean_selected_coordinate_weight"] = (
            summary["coordinate_weight_sum"] / selected_coordinates
            if selected_coordinates
            else float("nan")
        )
        summary["gate_coverage_range"] = [0.05, 0.95]
        if protocol["distill_stage"] == "refined":
            summary["selected_tiny_roi_coverage"] = (
                summary["selected_tiny_rois"] / summary["positive_tiny_rois"]
                if summary["positive_tiny_rois"]
                else float("nan")
            )
            summary["gate_conflict_rate_max"] = 0.10
            summary["gate_norm_ratio_range"] = [0.02, 0.20]
            summary["gate_tiny_roi_coverage_min"] = 0.10
            summary["gate_pass"] = bool(
                complete
                and overall["valid_batches"] >= protocol["batches"]
                and 0.05 <= summary["selected_coordinate_coverage"] <= 0.95
                and summary["coordinate_weight_sum"] > 0
                and overall["conflict_rate"] <= 0.10
                and 0.02 <= overall["norm_ratio_mean"] <= 0.20
                and overall["cosine_mean"] > 0
                and summary["selected_tiny_roi_coverage"] >= 0.10
            )
        elif protocol["cross_head"]:
            summary["selected_tiny_roi_coverage"] = (
                summary["selected_tiny_rois"] / summary["positive_tiny_rois"]
                if summary["positive_tiny_rois"]
                else float("nan")
            )
            summary["gate_conflict_rate_max"] = 0.10
            summary["gate_norm_ratio_range"] = [0.03, 0.20]
            summary["gate_tiny_roi_coverage_min"] = 0.10
            summary["gate_pass"] = bool(
                complete
                and overall["valid_batches"] >= protocol["batches"]
                and 0.05 <= summary["selected_coordinate_coverage"] <= 0.95
                and summary["coordinate_weight_sum"] > 0
                and overall["conflict_rate"] < 0.10
                and 0.03 <= overall["norm_ratio_mean"] <= 0.20
                and overall["cosine_mean"] > 0
                and summary["selected_tiny_roi_coverage"] >= 0.10
            )
        elif protocol["distill_distance"] == "ordered_w1":
            summary["gate_conflict_rate_max"] = 0.05
            summary["gate_norm_ratio_range"] = [0.02, 0.20]
            summary["gate_cosine_reference_floor"] = (
                protocol["reference_kl_cosine_mean"] - 0.02)
            summary["gate_pass"] = bool(
                complete
                and overall["valid_batches"] >= protocol["batches"]
                and 0.05 <= summary["selected_coordinate_coverage"] <= 0.95
                and summary["coordinate_weight_sum"] > 0
                and overall["conflict_rate"] <= 0.05
                and 0.02 <= overall["norm_ratio_mean"] <= 0.20
                and overall["cosine_mean"] > 0
                and overall["cosine_mean"]
                >= summary["gate_cosine_reference_floor"]
            )
        else:
            summary["gate_pass"] = bool(
                complete
                and overall["valid_batches"] >= protocol["batches"]
                and 0.05 <= summary["selected_coordinate_coverage"] <= 0.95
                and summary["coordinate_weight_sum"] > 0
            )
    else:
        summary["gate_threshold"] = 0.10
        summary["gate_pass"] = bool(
            complete
            and overall["valid_batches"] >= protocol["batches"]
            and overall["conflict_rate"] >= summary["gate_threshold"]
        )
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


def _move_batch(images, targets):
    images = [image.to(DEVICE) for image in images]
    targets = [
        {
            key: value.to(DEVICE) if isinstance(value, torch.Tensor) else value
            for key, value in target.items()
        }
        for target in targets
    ]
    return images, targets


def main() -> None:
    args = parse_args()
    if args.batches <= 0 or args.batch_size <= 0:
        raise ValueError("batches and batch-size must be positive")
    if args.num_workers < 0 or args.save_every <= 0 or args.band_every <= 0:
        raise ValueError(
            "num-workers must be non-negative; save-every and band-every "
            "must be positive"
        )
    teacher_path = args.teacher.resolve()
    if not teacher_path.exists():
        raise FileNotFoundError(teacher_path)
    reference_kl_cosine_mean = None
    if args.distill_distance == "ordered_w1":
        if not args.coordinate_reliable:
            raise ValueError("Ordered-W1 audit requires coordinate reliability")
        if not args.reference_json.is_file():
            raise FileNotFoundError(args.reference_json)
        reference_artifact = json.loads(
            args.reference_json.read_text(encoding="utf-8"))
        reference_kl_cosine_mean = float(
            reference_artifact["summary"]["overall"]["cosine_mean"])
    if args.cross_head and not args.coordinate_reliable:
        raise ValueError("Cross-head audit requires coordinate reliability")

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
        transform_min_sizes=(640,),
        transform_max_size=800,
    ).to(DEVICE)
    checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
    teacher = deepcopy(student)
    teacher.load_state_dict(checkpoint["model"])
    teacher.to(DEVICE)
    attach_cbl_cross_scale_teacher(
        student,
        teacher,
        loss_weight=0.25,
        temperature=2.0,
        advantage_margin=0.02,
        teacher_min_size=960,
        teacher_max_size=1200,
        tiny_reference=16.0,
        tiny_weight_cap=2.0,
        coordinate_reliable=args.coordinate_reliable,
        distill_distance=args.distill_distance,
        cross_head=args.cross_head,
        distill_stage=args.distill_stage,
    )
    student.roi_heads._cbl_scale_collect_audit = True
    student.train()
    teacher.eval()
    assert not any(
        "_cbl_scale_teacher" in key for key in student.state_dict())
    shared_parameters = (
        tuple(student.roi_heads.box_head.parameters())
        if args.shared_parameters == "box_head"
        else tuple(chain(
            student.backbone.parameters(),
            student.roi_heads.box_head.parameters(),
        ))
    )
    shared_parameters = tuple(
        parameter for parameter in shared_parameters
        if parameter.requires_grad
    )
    if not shared_parameters:
        raise RuntimeError("RoI box head has no shared parameters")

    amp_enabled = DEVICE.type == "cuda" and not args.no_amp
    protocol = {
        "seed": args.seed,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "band_every": args.band_every,
        "weighted_sampler": True,
        "copy_paste": bool(copy_paste_pool),
        "student_initialization": "torchvision FasterRCNN ResNet50-FPN DEFAULT",
        "student_scale": [640, 800],
        "teacher_path": str(teacher_path),
        "teacher_epoch": checkpoint.get("epoch"),
        "teacher_model_source": checkpoint.get("model_source"),
        "teacher_scale": [960, 1200],
        "distillation_weight": 0.25,
        "coordinate_reliable": args.coordinate_reliable,
        "distill_distance": args.distill_distance,
        "reference_json": (
            str(args.reference_json.resolve())
            if args.distill_distance == "ordered_w1"
            else None
        ),
        "reference_kl_cosine_mean": reference_kl_cosine_mean,
        "cross_head": args.cross_head,
        "distill_stage": args.distill_stage,
        "temperature": 2.0,
        "advantage_margin": 0.02,
        "tiny_cutoff_px": 16.0,
        "shared_parameters": args.shared_parameters,
        "detector_loss": "sum(all returned losses except loss_box_scale_distill)",
        "amp": amp_enabled,
        "optimizer_updates": 0,
    }
    del checkpoint

    rows: list[dict] = []
    started = time.time()
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images, targets = _move_batch(images, targets)
        student.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            losses = student(images, targets)
            scale_loss = losses[DISTILL_LOSS_KEY]
            detector_loss = sum(
                loss for name, loss in losses.items() if name != DISTILL_LOSS_KEY
            )
        audit = student.roi_heads._cbl_scale_audit
        if audit is None:
            raise RuntimeError("Cross-scale audit telemetry was not collected")
        audit_bands = batch_index % args.band_every == 0
        gradient_losses = [
            ("detector", detector_loss),
            ("overall", scale_loss),
        ]
        if audit_bands:
            gradient_losses.extend([
                ("tiny", audit["tiny_loss"]),
                ("larger", audit["larger_loss"]),
            ])
        gradients = {}
        for position, (name, loss) in enumerate(gradient_losses):
            gradients[name] = torch.autograd.grad(
                loss,
                shared_parameters,
                retain_graph=position < len(gradient_losses) - 1,
                allow_unused=True,
            )
        row = {
            "batch": batch_index,
            "detector_loss": float(detector_loss.detach()),
            "scale_loss": float(scale_loss.detach()),
            "tiny_scale_loss": float(audit["tiny_loss"].detach()),
            "larger_scale_loss": float(audit["larger_loss"].detach()),
            "positive_rois": audit["positive_rois"],
            "selected_rois": audit["selected_rois"],
            "positive_coordinates": audit["positive_coordinates"],
            "selected_coordinates": audit["selected_coordinates"],
            "coordinate_weight_sum": audit["coordinate_weight_sum"],
            "selected_tiny_rois": audit["selected_tiny_rois"],
            "selected_larger_rois": audit["selected_larger_rois"],
            "positive_tiny_rois": audit["positive_tiny_rois"],
            "positive_larger_rois": audit["positive_larger_rois"],
            "bands_audited": audit_bands,
            "overall": _gradient_pair_stats(
                gradients["detector"], gradients["overall"]),
            "tiny": (
                _gradient_pair_stats(gradients["detector"], gradients["tiny"])
                if audit_bands
                else _not_audited_stats()
            ),
            "larger": (
                _gradient_pair_stats(
                    gradients["detector"], gradients["larger"])
                if audit_bands
                else _not_audited_stats()
            ),
        }
        row["overall"]["raw_norm_ratio"] = (
            row["overall"]["norm_ratio"] / 0.25
        )
        rows.append(row)
        assert all(parameter.grad is None for parameter in teacher.parameters())
        student.roi_heads._cbl_scale_audit = None
        if batch_index % args.save_every == 0 or batch_index == args.batches:
            elapsed = time.time() - started
            _write_artifact(
                args.out_json,
                protocol,
                rows,
                complete=batch_index == args.batches,
                elapsed_seconds=elapsed,
            )
            summary = _summarize(rows, "overall")
            print(
                f"[{batch_index:03d}/{args.batches}] "
                f"conflict={summary['conflict_rate']:.3f}, "
                f"cos={summary['cosine_mean']:.4f}, "
                f"ratio={summary['norm_ratio_mean']:.4f}, "
                f"elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

    artifact = json.loads(args.out_json.read_text(encoding="utf-8"))
    artifact["peak_allocated_gib"] = (
        torch.cuda.max_memory_allocated() / 1024**3
        if DEVICE.type == "cuda"
        else 0.0
    )
    args.out_json.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact["summary"], indent=2))
    print(f"Saved audit: {args.out_json}")


if __name__ == "__main__":
    main()
