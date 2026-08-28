"""Evaluate cached horizontal-flip TTA fusion variants."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.dataset import YOLOTinyDataset
from common.eval_utils import (
    compute_class_aware_scale_ap,
    compute_scale_ap,
    evaluate_coco,
)
from scripts.eval_cbl_flip_tta import (
    _audit_predictions,
    _finalize,
    _greedy_cross_view_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate offline variants from a flip-TTA prediction cache"
    )
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument("--split", choices=("valid",), default="valid")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size-cutoffs", default="8,12,16")
    parser.add_argument("--detections-per-image", type=int, default=200)
    return parser.parse_args()


def _box_sizes(boxes: torch.Tensor) -> torch.Tensor:
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0.0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0.0)
    return (widths * heights).sqrt()


def _pair_fuse_tiny_gate(
    original: dict[str, torch.Tensor],
    flipped: dict[str, torch.Tensor],
    *,
    size_cutoff: float,
    tiny_mode: str,
    detections_per_image: int,
) -> tuple[dict[str, torch.Tensor], dict[str, int | float]]:
    matches, _ = _greedy_cross_view_pairs(original, flipped, 0.50)
    original_sizes = _box_sizes(original["boxes"])
    boxes, scores, labels = [], [], []
    matched_pairs = 0
    gated_pairs = 0

    for original_index in range(len(original["boxes"])):
        box = original["boxes"][original_index]
        score = original["scores"][original_index]
        flipped_index = matches.get(original_index)
        if flipped_index is not None:
            matched_pairs += 1
            flipped_box = flipped["boxes"][flipped_index]
            flipped_score = flipped["scores"][flipped_index]
            is_tiny = float(original_sizes[original_index]) < size_cutoff
            if is_tiny:
                gated_pairs += 1
                if tiny_mode == "keep_box":
                    score = (score + flipped_score) / 2
                elif tiny_mode == "keep_original":
                    pass
                else:
                    raise ValueError(f"Unknown tiny mode: {tiny_mode}")
            else:
                weights = torch.stack((score, flipped_score)).clamp(min=1e-8)
                box = (weights[0] * box + weights[1] * flipped_box) / weights.sum()
                score = (score + flipped_score) / 2
        boxes.append(box)
        scores.append(score)
        labels.append(original["labels"][original_index])

    if not boxes:
        fused = {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }
    else:
        fused = _finalize(
            torch.stack(boxes),
            torch.stack(scores),
            torch.stack(labels),
            nms_threshold=0.5,
            detections_per_image=detections_per_image,
        )
    stats = {
        "size_cutoff": size_cutoff,
        "matched_pairs": matched_pairs,
        "gated_pairs": gated_pairs,
        "gated_rate_vs_matched": round(
            gated_pairs / matched_pairs, 6
        )
        if matched_pairs
        else 0.0,
    }
    return fused, stats


def _subset(items: list[dict[str, torch.Tensor]], parity: int) -> list[dict[str, torch.Tensor]]:
    return [item for index, item in enumerate(items) if index % 2 == parity]


def _metrics_bundle(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
) -> dict:
    return {
        **evaluate_coco(predictions, targets, class_metrics=False),
        **compute_scale_ap(predictions, targets),
        **compute_class_aware_scale_ap(predictions, targets),
    }


def main() -> None:
    args = parse_args()
    cache_path = (
        args.prediction_cache
        if args.prediction_cache.is_absolute()
        else ROOT / args.prediction_cache
    )
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    originals = cache["original"]
    flipped = cache["flip"]
    targets = cache["targets"]
    selected = cache["selected"]
    cutoffs = [
        float(value.strip())
        for value in args.size_cutoffs.split(",")
        if value.strip()
    ]

    predictions = {
        "original": originals,
        "selected_pair_score_weighted_mean_iou50": selected,
    }
    pair_stats: dict[str, dict] = {}
    for cutoff in cutoffs:
        cutoff_label = f"{int(cutoff):02d}" if cutoff.is_integer() else str(cutoff)
        for tiny_mode in ("keep_box", "keep_original"):
            name = f"tiny_{tiny_mode}_lt{cutoff_label}_pair_iou50"
            fused_predictions = []
            stats_total = {
                "size_cutoff": cutoff,
                "matched_pairs": 0,
                "gated_pairs": 0,
            }
            for original, flip in zip(originals, flipped):
                fused, stats = _pair_fuse_tiny_gate(
                    original,
                    flip,
                    size_cutoff=cutoff,
                    tiny_mode=tiny_mode,
                    detections_per_image=args.detections_per_image,
                )
                fused_predictions.append(fused)
                stats_total["matched_pairs"] += int(stats["matched_pairs"])
                stats_total["gated_pairs"] += int(stats["gated_pairs"])
            stats_total["gated_rate_vs_matched"] = round(
                stats_total["gated_pairs"] / stats_total["matched_pairs"], 6
            )
            predictions[name] = fused_predictions
            pair_stats[name] = stats_total

    metrics = {
        name: _metrics_bundle(prediction, targets)
        for name, prediction in predictions.items()
    }
    leader_names = {
        "coco_AP": max(metrics, key=lambda name: metrics[name]["coco_AP"]),
        "coco_AP75": max(metrics, key=lambda name: metrics[name]["coco_AP75"]),
        "coco_AR100": max(metrics, key=lambda name: metrics[name]["coco_AR100"]),
    }
    fold_metrics = {}
    for parity in (0, 1):
        fold_targets = _subset(targets, parity)
        fold_metrics[str(parity)] = {
            "tiles": len(fold_targets),
            **{
                name: evaluate_coco(
                    _subset(prediction, parity),
                    fold_targets,
                    class_metrics=False,
                )
                for name, prediction in predictions.items()
            },
        }

    data_dir = ROOT / "data" / args.split
    dataset = YOLOTinyDataset(
        img_dir=data_dir / "images",
        lbl_dir=data_dir / "labels",
        is_train=False,
    )
    audit_names = sorted(set(leader_names.values()) | {"original", "selected_pair_score_weighted_mean_iou50"})
    ap75_audit = {
        name: _audit_predictions(predictions[name], targets, dataset)
        for name in audit_names
    }

    summary = {
        "prediction_cache": str(cache_path),
        "checkpoint": cache.get("checkpoint"),
        "profile": cache.get("profile"),
        "split": args.split,
        "tiles": len(targets),
        "size_cutoffs": cutoffs,
        "metrics": metrics,
        "leader_names": leader_names,
        "pair_stats": pair_stats,
        "fold_metrics": fold_metrics,
        "ap75_audit": ap75_audit,
    }
    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
