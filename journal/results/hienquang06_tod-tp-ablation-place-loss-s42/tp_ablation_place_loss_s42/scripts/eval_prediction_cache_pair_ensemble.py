"""Evaluate pair/union ensembles between two prediction caches."""
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
        description="Evaluate cached prediction-pair ensembles"
    )
    parser.add_argument("--cache-a", type=Path, required=True)
    parser.add_argument("--key-a", required=True)
    parser.add_argument("--name-a", default="a")
    parser.add_argument("--cache-b", type=Path, required=True)
    parser.add_argument("--key-b", required=True)
    parser.add_argument("--name-b", default="b")
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


def _load_predictions(cache_path: Path, key: str) -> tuple[dict, list[dict]]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    if key not in cache:
        available = ", ".join(sorted(cache))
        raise KeyError(f"Key {key!r} not in {cache_path}; available: {available}")
    return cache, cache[key]


def main() -> None:
    args = parse_args()
    cache_a_path = _resolve(args.cache_a)
    cache_b_path = _resolve(args.cache_b)
    cache_a, preds_a = _load_predictions(cache_a_path, args.key_a)
    cache_b, preds_b = _load_predictions(cache_b_path, args.key_b)
    targets = cache_a.get("targets")
    if targets is None:
        raise KeyError(f"Cache {cache_a_path} has no targets")
    if len(preds_a) != len(preds_b) or len(preds_a) != len(targets):
        raise ValueError("Prediction/target lengths do not match")

    predictions: dict[str, list[dict[str, torch.Tensor]]] = {
        args.name_a: preds_a,
        args.name_b: preds_b,
    }
    pair_stats = {}
    for threshold in (0.50, 0.60, 0.70):
        pair_name = f"pair_score_weighted_mean_iou{int(threshold * 100):02d}"
        predictions[pair_name] = []
        matched_pairs = 0
        a_detections = 0
        b_detections = 0
        for pred_a, pred_b in zip(preds_a, preds_b):
            pairing = _greedy_cross_view_pairs(pred_a, pred_b, threshold)
            matched_pairs += len(pairing[0])
            a_detections += len(pred_a["boxes"])
            b_detections += len(pred_b["boxes"])
            predictions[pair_name].append(
                _pair_fuse(
                    pred_a,
                    pred_b,
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
            f"{args.name_a}_detections": a_detections,
            f"{args.name_b}_detections": b_detections,
            f"match_rate_vs_{args.name_a}": round(
                matched_pairs / a_detections, 6
            ),
        }
    for threshold in (0.50, 0.60):
        name = f"union_nms{int(threshold * 100):02d}"
        predictions[name] = [
            _union_nms(
                pred_a,
                pred_b,
                threshold,
                args.detections_per_image,
            )
            for pred_a, pred_b in zip(preds_a, preds_b)
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
    audit_names = sorted(
        set(leader_names.values()) | {args.name_a, args.name_b}
    )
    ap75_audit = {
        name: _audit_predictions(predictions[name], targets, dataset)
        for name in audit_names
    }

    summary = {
        "cache_a": str(cache_a_path),
        "key_a": args.key_a,
        "name_a": args.name_a,
        "cache_b": str(cache_b_path),
        "key_b": args.key_b,
        "name_b": args.name_b,
        "profile_a": cache_a.get("profile"),
        "profile_b": cache_b.get("profile"),
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
