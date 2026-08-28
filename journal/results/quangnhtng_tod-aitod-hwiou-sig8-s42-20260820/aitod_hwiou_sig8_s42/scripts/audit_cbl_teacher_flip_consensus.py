"""Audit whether teacher flip agreement predicts reliable CBL coordinates."""
from __future__ import annotations

import argparse
import hashlib
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
from common.model import attach_cbl_cross_scale_teacher, build_model


DEFAULT_TEACHER = (
    ROOT
    / ".runtime/kaggle/cbl_iterative_train_fair20/output/tod_output/runs"
    / "sa_alw_full__cbl__irtw0.5ir1s0.3__la_loss__seed42__cbl_iterative_train_fair20"
    / "best.pt"
)
DEFAULT_OUTPUT = ROOT / "runs/cf_cr_sc_cbl_consensus_audit_seed42.json"


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


def _model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _binary_auc(scores: list[float], labels: list[bool]) -> float:
    if len(scores) != len(labels) or not scores:
        return float("nan")
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ordered = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and scores[ordered[end]] == scores[ordered[index]]:
            end += 1
        average_rank = 0.5 * ((index + 1) + end)
        rank_sum += average_rank * sum(labels[item] for item in ordered[index:end])
        index = end
    return (
        rank_sum - positives * (positives + 1) / 2
    ) / (positives * negatives)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def _write_artifact(
    path: Path,
    *,
    protocol: dict,
    rows: list[dict],
    samples: dict[str, list[float]],
    complete: bool,
    elapsed_seconds: float,
    summary: dict | None = None,
) -> None:
    artifact = {
        "complete": complete,
        "protocol": protocol,
        "elapsed_seconds": elapsed_seconds,
        "rows": rows,
        "samples": samples,
        "summary": summary,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.batches <= 0 or args.batch_size <= 0 or args.save_every <= 0:
        raise ValueError("batches, batch-size, and save-every must be positive")
    if args.num_workers < 0 or not args.teacher.is_file():
        raise ValueError("num-workers must be non-negative and teacher must exist")

    seed_all(args.seed)
    dataset = build_training_datasets(use_patches=False, is_train=True)
    copy_pool = build_copy_paste_pool(dataset)
    if copy_pool:
        dataset.copy_paste_pool = copy_pool
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
    reliability = compute_reliability_threshold(dataset)
    student = build_model(
        metric_fn=get_metric_fn("sa_alw_full"),
        placement="la_loss",
        reliability_thr=reliability,
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
    checkpoint = torch.load(args.teacher, map_location="cpu", weights_only=False)
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
        coordinate_reliable=True,
        consensus_filter=True,
    )
    student.roi_heads._cbl_scale_collect_audit = True
    student.train()
    teacher.eval()
    if any("_cbl_scale_teacher" in key for key in student.state_dict()):
        raise RuntimeError("Teacher leaked into student state dict")
    initial_state_sha256 = _model_state_sha256(student)
    amp_enabled = DEVICE.type == "cuda" and not args.no_amp
    protocol = {
        "method": "Consensus-Filtered CR-SC-CBL",
        "seed": args.seed,
        "batches": args.batches,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "weighted_sampler": True,
        "copy_paste": bool(copy_pool),
        "student_scale": [640, 800],
        "teacher_scale": [960, 1200],
        "teacher_path": str(args.teacher.resolve()),
        "teacher_epoch": checkpoint.get("epoch"),
        "teacher_model_source": checkpoint.get("model_source"),
        "teacher_views": ["original", "horizontal_flip"],
        "consensus": "1 - JS(raw-temperature teacher views) / log(2)",
        "optimizer_updates": 0,
        "backward_calls": 0,
        "validation_access": False,
        "locked_test_access": False,
        "gates": {
            "finite_batches": args.batches,
            "retained_weight_ratio_min": 0.20,
            "tiny_retained_weight_ratio_min": 0.10,
            "agreement_std_min": 0.01,
            "disagreement_high_error_auc_min": 0.55,
            "low_vs_high_agreement_error_ratio_min": 1.05,
            "flip_alignment_max_error_max": 1e-3,
        },
    }
    del checkpoint

    rows: list[dict] = []
    samples = {
        "agreement": [],
        "teacher_error": [],
        "flip_teacher_error": [],
        "base_weight": [],
        "target_size": [],
    }
    started = time.time()
    finite_batches = 0
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = [image.to(DEVICE) for image in images]
        targets = [
            {
                key: value.to(DEVICE) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            }
            for target in targets
        ]
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=amp_enabled):
            losses = student(images, targets)
        audit = student.roi_heads._cbl_scale_audit
        if audit is None:
            raise RuntimeError("Consensus telemetry was not collected")
        loss_finite = all(torch.isfinite(loss) for loss in losses.values())
        finite_batches += int(loss_finite)
        base_weights = audit.pop("_base_coordinate_weights").float().cpu()
        agreements = audit.pop("_consensus_agreements").float().cpu()
        teacher_errors = audit.pop("_teacher_coordinate_errors").float().cpu()
        flip_errors = audit.pop("_flip_coordinate_errors").float().cpu()
        target_sizes = audit.pop("_target_sizes").float().cpu()
        selected = base_weights > 0
        expanded_sizes = target_sizes.unsqueeze(-1).expand_as(base_weights)
        samples["agreement"].extend(agreements[selected].tolist())
        samples["teacher_error"].extend(teacher_errors[selected].tolist())
        samples["flip_teacher_error"].extend(flip_errors[selected].tolist())
        samples["base_weight"].extend(base_weights[selected].tolist())
        samples["target_size"].extend(expanded_sizes[selected].tolist())
        rows.append({
            "batch": batch_index,
            "finite": bool(loss_finite),
            "positive_rois": audit["positive_rois"],
            "selected_rois": audit["selected_rois"],
            "positive_coordinates": audit["positive_coordinates"],
            "selected_coordinates": audit["selected_coordinates"],
            "base_coordinate_weight_sum": audit["base_coordinate_weight_sum"],
            "coordinate_weight_sum": audit["coordinate_weight_sum"],
            "retained_weight_ratio": audit["consensus_retained_weight_ratio"],
            "agreement_mean": audit["consensus_agreement_mean"],
            "flip_alignment_max_error": audit["flip_alignment_max_error"],
        })
        if any(parameter.grad is not None for parameter in teacher.parameters()):
            raise RuntimeError("Frozen teacher received a gradient")
        student.roi_heads._cbl_scale_audit = None
        if batch_index % args.save_every == 0 or batch_index == args.batches:
            elapsed = time.time() - started
            _write_artifact(
                args.out_json,
                protocol=protocol,
                rows=rows,
                samples=samples,
                complete=False,
                elapsed_seconds=elapsed,
            )
            print(
                f"[{batch_index:03d}/{args.batches}] "
                f"samples={len(samples['agreement'])}, "
                f"retain={statistics.fmean(row['retained_weight_ratio'] for row in rows):.4f}, "
                f"elapsed={elapsed / 60:.1f}m",
                flush=True,
            )

    final_state_sha256 = _model_state_sha256(student)
    agreements = samples["agreement"]
    errors = samples["teacher_error"]
    weights = samples["base_weight"]
    sizes = samples["target_size"]
    if not agreements:
        raise RuntimeError("No CR-selected coordinates were audited")
    median_error = statistics.median(errors)
    disagreement_auc = _binary_auc(
        [1.0 - value for value in agreements],
        [value > median_error for value in errors],
    )
    paired = sorted(zip(agreements, errors), key=lambda item: item[0])
    quartile_count = max(1, len(paired) // 4)
    low_error = statistics.fmean(value for _, value in paired[:quartile_count])
    high_error = statistics.fmean(value for _, value in paired[-quartile_count:])
    low_high_ratio = low_error / max(high_error, 1e-12)
    retained_weight_ratio = sum(
        weight * agreement for weight, agreement in zip(weights, agreements)
    ) / max(sum(weights), 1e-12)
    tiny = [size < 16.0 for size in sizes]
    tiny_weight_sum = sum(weight for weight, flag in zip(weights, tiny) if flag)
    tiny_retained_weight_ratio = sum(
        weight * agreement
        for weight, agreement, flag in zip(weights, agreements, tiny)
        if flag
    ) / max(tiny_weight_sum, 1e-12)
    alignment_max_error = max(row["flip_alignment_max_error"] for row in rows)
    summary = {
        "finite_batches": finite_batches,
        "selected_coordinate_samples": len(agreements),
        "agreement_mean": statistics.fmean(agreements),
        "agreement_std": statistics.pstdev(agreements),
        "agreement_q1": _quantile(agreements, 0.25),
        "agreement_median": _quantile(agreements, 0.50),
        "agreement_q3": _quantile(agreements, 0.75),
        "retained_weight_ratio": retained_weight_ratio,
        "tiny_retained_weight_ratio": tiny_retained_weight_ratio,
        "disagreement_high_error_auc": disagreement_auc,
        "low_agreement_teacher_error": low_error,
        "high_agreement_teacher_error": high_error,
        "low_vs_high_agreement_error_ratio": low_high_ratio,
        "flip_alignment_max_error": alignment_max_error,
        "student_state_sha256_before": initial_state_sha256,
        "student_state_sha256_after": final_state_sha256,
        "student_state_unchanged": initial_state_sha256 == final_state_sha256,
    }
    gates = protocol["gates"]
    summary["gate_pass"] = bool(
        finite_batches == args.batches
        and retained_weight_ratio >= gates["retained_weight_ratio_min"]
        and tiny_retained_weight_ratio >= gates["tiny_retained_weight_ratio_min"]
        and summary["agreement_std"] >= gates["agreement_std_min"]
        and disagreement_auc >= gates["disagreement_high_error_auc_min"]
        and low_high_ratio >= gates["low_vs_high_agreement_error_ratio_min"]
        and alignment_max_error <= gates["flip_alignment_max_error_max"]
        and summary["student_state_unchanged"]
    )
    elapsed = time.time() - started
    _write_artifact(
        args.out_json,
        protocol=protocol,
        rows=rows,
        samples=samples,
        complete=True,
        elapsed_seconds=elapsed,
        summary=summary,
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved audit: {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
