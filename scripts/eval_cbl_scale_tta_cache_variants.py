"""Evaluate cached transform-scale TTA fusion variants."""
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
    _greedy_cross_view_pairs,
    _pair_fuse,
    _union_nms,
)
from scripts.eval_cbl_scale_tta import _audit_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scale-TTA variants from cached predictions"
    )
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument("--split", choices=("valid",), default="valid")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--detections-per-image", type=int, default=200)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _subset(
    items: list[dict[str, torch.Tensor]],
    parity: int,
) -> list[dict[str, torch.Tensor]]:
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


def _same_class_score_nms(
    first: dict[str, torch.Tensor],
    second: dict[str, torch.Tensor],
    threshold: float,
    detections_per_image: int,
) -> dict[str, torch.Tensor]:
    return _union_nms(first, second, threshold, detections_per_image)


def main() -> None:
    args = parse_args()
    cache_path = _resolve(args.prediction_cache)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    base_predictions = cache["base"]
    scale_predictions = cache["scale"]
    targets = cache["targets"]

    predictions: dict[str, list[dict[str, torch.Tensor]]] = {
        "base": base_predictions,
        "scale": scale_predictions,
    }
    pair_stats = {}
    for threshold in (0.50, 0.60, 0.70):
        pair_name = f"pair_score_weighted_mean_iou{int(threshold * 100):02d}"
        predictions[pair_name] = []
        matched_pairs = 0
        base_detections = 0
        scale_detections = 0
        for base, scale in zip(base_predictions, scale_predictions):
            pairing = _greedy_cross_view_pairs(base, scale, threshold)
            matched_pairs += len(pairing[0])
            base_detections += len(base["boxes"])
            scale_detections += len(scale["boxes"])
            predictions[pair_name].append(
                _pair_fuse(
                    base,
                    scale,
                    pair_threshold=threshold,
                    coordinate_mode="score_weighted",
                    score_mode="mean",
                    include_unmatched_flip=False,
                    unmatched_flip_weight=1.0,
                    detections_per_image=args.detections_per_image,
                    pairing=pairing,
                )
            )
        pair_stats[pair_name] = {
            "matched_pairs": matched_pairs,
            "base_detections": base_detections,
            "scale_detections": scale_detections,
            "match_rate_vs_base": round(matched_pairs / base_detections, 6),
        }
    unmatched_variants = [
        (0.50, 0.25),
        (0.50, 0.50),
        (0.50, 0.75),
        (0.50, 0.90),
        (0.45, 0.50),
        (0.55, 0.50),
    ]
    for threshold, weight in unmatched_variants:
        pair_name = (
            f"pair_with_unmatched_scale_iou{int(threshold * 100):02d}"
            f"_w{int(weight * 100):02d}"
        )
        predictions[pair_name] = []
        matched_pairs = 0
        base_detections = 0
        scale_detections = 0
        for base, scale in zip(base_predictions, scale_predictions):
            pairing = _greedy_cross_view_pairs(base, scale, threshold)
            matched_pairs += len(pairing[0])
            base_detections += len(base["boxes"])
            scale_detections += len(scale["boxes"])
            predictions[pair_name].append(
                _pair_fuse(
                    base,
                    scale,
                    pair_threshold=threshold,
                    coordinate_mode="score_weighted",
                    score_mode="mean",
                    include_unmatched_flip=True,
                    unmatched_flip_weight=weight,
                    detections_per_image=args.detections_per_image,
                    pairing=pairing,
                )
            )
        pair_stats[pair_name] = {
            "matched_pairs": matched_pairs,
            "base_detections": base_detections,
            "scale_detections": scale_detections,
            "match_rate_vs_base": round(matched_pairs / base_detections, 6),
            "unmatched_scale_score_weight": weight,
        }
    for threshold in (0.50, 0.60):
        name = f"union_nms{int(threshold * 100):02d}"
        predictions[name] = [
            _same_class_score_nms(
                base,
                scale,
                threshold,
                args.detections_per_image,
            )
            for base, scale in zip(base_predictions, scale_predictions)
        ]

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
    audit_names = sorted(set(leader_names.values()) | {"base", "scale"})
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
