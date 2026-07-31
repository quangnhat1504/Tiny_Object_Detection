"""Evaluate cached ensembles between scalar and strict flip-TTA profiles."""
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
    _pair_fuse,
    _union_nms,
)
from scripts.eval_cbl_flip_tta_cache_variants import _pair_fuse_tiny_gate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scalar/strict cached flip-TTA profile ensembles"
    )
    parser.add_argument("--scalar-cache", type=Path, required=True)
    parser.add_argument("--strict-cache", type=Path, required=True)
    parser.add_argument("--split", choices=("valid",), default="valid")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--strict-tiny-cutoff", type=float, default=12.0)
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


def main() -> None:
    args = parse_args()
    scalar_path = _resolve(args.scalar_cache)
    strict_path = _resolve(args.strict_cache)
    scalar_cache = torch.load(scalar_path, map_location="cpu", weights_only=False)
    strict_cache = torch.load(strict_path, map_location="cpu", weights_only=False)

    targets = scalar_cache["targets"]
    strict_tiny = []
    strict_tiny_stats = {
        "matched_pairs": 0,
        "gated_pairs": 0,
        "strict_tiny_cutoff": args.strict_tiny_cutoff,
    }
    for original, flip in zip(strict_cache["original"], strict_cache["flip"]):
        fused, stats = _pair_fuse_tiny_gate(
            original,
            flip,
            size_cutoff=args.strict_tiny_cutoff,
            tiny_mode="keep_box",
            detections_per_image=args.detections_per_image,
        )
        strict_tiny.append(fused)
        strict_tiny_stats["matched_pairs"] += int(stats["matched_pairs"])
        strict_tiny_stats["gated_pairs"] += int(stats["gated_pairs"])
    strict_tiny_stats["gated_rate_vs_matched"] = round(
        strict_tiny_stats["gated_pairs"] / strict_tiny_stats["matched_pairs"],
        6,
    )

    scalar_selected = scalar_cache["selected"]
    predictions = {
        "scalar_selected": scalar_selected,
        "strict_tiny_keep_box_lt12": strict_tiny,
        "scalar_strict_pair_mean_iou50": [],
        "scalar_strict_pair_mean_iou60": [],
        "scalar_strict_union_nms050": [],
        "scalar_strict_union_nms060": [],
    }
    for scalar_pred, strict_pred in zip(scalar_selected, strict_tiny):
        predictions["scalar_strict_pair_mean_iou50"].append(
            _pair_fuse(
                scalar_pred,
                strict_pred,
                pair_threshold=0.50,
                coordinate_mode="score_weighted",
                score_mode="mean",
                include_unmatched_flip=False,
                unmatched_flip_weight=1.0,
                detections_per_image=args.detections_per_image,
            )
        )
        predictions["scalar_strict_pair_mean_iou60"].append(
            _pair_fuse(
                scalar_pred,
                strict_pred,
                pair_threshold=0.60,
                coordinate_mode="score_weighted",
                score_mode="mean",
                include_unmatched_flip=False,
                unmatched_flip_weight=1.0,
                detections_per_image=args.detections_per_image,
            )
        )
        predictions["scalar_strict_union_nms050"].append(
            _union_nms(
                scalar_pred,
                strict_pred,
                0.50,
                args.detections_per_image,
            )
        )
        predictions["scalar_strict_union_nms060"].append(
            _union_nms(
                scalar_pred,
                strict_pred,
                0.60,
                args.detections_per_image,
            )
        )

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
    audit_names = sorted(
        set(leader_names.values())
        | {"scalar_selected", "strict_tiny_keep_box_lt12"}
    )
    ap75_audit = {
        name: _audit_predictions(predictions[name], targets, dataset)
        for name in audit_names
    }

    summary = {
        "scalar_cache": str(scalar_path),
        "strict_cache": str(strict_path),
        "split": args.split,
        "tiles": len(targets),
        "strict_tiny_stats": strict_tiny_stats,
        "metrics": metrics,
        "leader_names": leader_names,
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
