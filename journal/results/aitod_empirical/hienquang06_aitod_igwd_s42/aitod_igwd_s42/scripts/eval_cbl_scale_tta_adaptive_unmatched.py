"""Tune adaptive unmatched-scale fusion from cached CBL predictions."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

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
)
from scripts.eval_cbl_scale_tta import _audit_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune size/score-aware unmatched-scale fusion on one image-group "
            "fold and confirm shortlisted rules on the other fold"
        )
    )
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument("--split", choices=("valid",), default="valid")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-predictions", type=Path, default=None)
    parser.add_argument("--pair-threshold", type=float, default=0.50)
    parser.add_argument("--detections-per-image", type=int, default=200)
    parser.add_argument("--shortlist-per-metric", type=int, default=3)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _candidate_rules() -> dict[str, dict[str, Any] | None]:
    rules: dict[str, dict[str, Any] | None] = {
        "pair_only": None,
        "unmatched_w75": {"weight": 0.75},
    }
    for score_floor in (0.05, 0.10, 0.15, 0.20):
        rules[f"unmatched_w75_score_ge{int(score_floor * 100):02d}"] = {
            "weight": 0.75,
            "score_floor": score_floor,
        }
    for minimum_size in (6.0, 8.0, 12.0, 16.0):
        rules[f"unmatched_w75_size_ge{int(minimum_size):02d}"] = {
            "weight": 0.75,
            "minimum_size": minimum_size,
        }
    for maximum_size in (8.0, 16.0, 24.0, 32.0):
        rules[f"unmatched_w75_size_lt{int(maximum_size):02d}"] = {
            "weight": 0.75,
            "maximum_size": maximum_size,
        }
    rules.update(
        {
            "unmatched_size_step08_w50_90": {
                "size_edges": [8.0],
                "size_weights": [0.50, 0.90],
            },
            "unmatched_size_step16_w50_90": {
                "size_edges": [16.0],
                "size_weights": [0.50, 0.90],
            },
            "unmatched_size_step16_w65_90": {
                "size_edges": [16.0],
                "size_weights": [0.65, 0.90],
            },
            "unmatched_size_band08_16_w85_50_90": {
                "size_edges": [8.0, 16.0],
                "size_weights": [0.85, 0.50, 0.90],
            },
            "unmatched_size_band06_16_w90_60_90": {
                "size_edges": [6.0, 16.0],
                "size_weights": [0.90, 0.60, 0.90],
            },
            "unmatched_score_step15_w50_90": {
                "score_cutoff": 0.15,
                "score_weights": [0.50, 0.90],
            },
            "unmatched_score_step20_w50_90": {
                "score_cutoff": 0.20,
                "score_weights": [0.50, 0.90],
            },
            "unmatched_score_step20_w65_90": {
                "score_cutoff": 0.20,
                "score_weights": [0.65, 0.90],
            },
            "unmatched_score_step30_w50_100": {
                "score_cutoff": 0.30,
                "score_weights": [0.50, 1.00],
            },
        }
    )
    return rules


def _rule_weights(
    scores: torch.Tensor,
    sizes: torch.Tensor,
    rule: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    weights = torch.full_like(scores, float(rule.get("weight", 1.0)))
    if "size_edges" in rule:
        edges = list(rule["size_edges"])
        band_weights = list(rule["size_weights"])
        if len(band_weights) != len(edges) + 1:
            raise ValueError("size_weights must have one more entry than size_edges")
        weights.fill_(float(band_weights[-1]))
        for edge, weight in reversed(list(zip(edges, band_weights[:-1]))):
            weights = torch.where(
                sizes < float(edge),
                torch.full_like(weights, float(weight)),
                weights,
            )
    if "score_cutoff" in rule:
        low_weight, high_weight = rule["score_weights"]
        weights = torch.where(
            scores < float(rule["score_cutoff"]),
            torch.full_like(weights, float(low_weight)),
            torch.full_like(weights, float(high_weight)),
        )

    keep = scores >= float(rule.get("score_floor", 0.0))
    keep &= sizes >= float(rule.get("minimum_size", 0.0))
    keep &= sizes < float(rule.get("maximum_size", float("inf")))
    keep &= weights > 0
    return weights, keep


def _adaptive_pair_fuse(
    base: dict[str, torch.Tensor],
    scale: dict[str, torch.Tensor],
    *,
    pairing: tuple[dict[int, int], set[int]],
    rule: dict[str, Any] | None,
    pair_threshold: float,
    detections_per_image: int,
) -> dict[str, torch.Tensor]:
    if rule is None:
        return _pair_fuse(
            base,
            scale,
            pair_threshold=pair_threshold,
            coordinate_mode="score_weighted",
            score_mode="mean",
            include_unmatched_flip=False,
            unmatched_flip_weight=1.0,
            detections_per_image=detections_per_image,
            pairing=pairing,
        )

    matches, used_scale = pairing
    scale_scores = scale["scores"].clone()
    sizes = (
        (scale["boxes"][:, 2] - scale["boxes"][:, 0]).clamp(min=0.0)
        * (scale["boxes"][:, 3] - scale["boxes"][:, 1]).clamp(min=0.0)
    ).sqrt()
    weights, keep = _rule_weights(scale_scores, sizes, rule)
    blocked_scale = set(used_scale)
    for scale_index in range(len(scale_scores)):
        if scale_index in used_scale:
            continue
        if bool(keep[scale_index]):
            scale_scores[scale_index] *= weights[scale_index]
        else:
            blocked_scale.add(scale_index)

    adjusted_scale = {**scale, "scores": scale_scores}
    return _pair_fuse(
        base,
        adjusted_scale,
        pair_threshold=pair_threshold,
        coordinate_mode="score_weighted",
        score_mode="mean",
        include_unmatched_flip=True,
        unmatched_flip_weight=1.0,
        detections_per_image=detections_per_image,
        pairing=(matches, blocked_scale),
    )


def _subset(
    items: list[dict[str, torch.Tensor]],
    indices: list[int],
) -> list[dict[str, torch.Tensor]]:
    return [items[index] for index in indices]


def _ranked_names(
    metrics: dict[str, dict[str, float]],
    metric_name: str,
    count: int,
) -> list[str]:
    return sorted(
        metrics,
        key=lambda name: metrics[name][metric_name],
        reverse=True,
    )[:count]


def _robust_leader(
    names: set[str],
    tune_metrics: dict[str, dict[str, float]],
    confirm_metrics: dict[str, dict[str, float]],
    full_metrics: dict[str, dict[str, float]],
    metric_name: str,
    baseline: str = "unmatched_w75",
) -> str:
    eligible = [
        name
        for name in names
        if tune_metrics[name][metric_name]
        >= tune_metrics[baseline][metric_name]
        and confirm_metrics[name][metric_name]
        >= confirm_metrics[baseline][metric_name]
    ]
    return max(
        eligible or [baseline],
        key=lambda name: full_metrics[name][metric_name],
    )


def main() -> None:
    args = parse_args()
    cache_path = _resolve(args.prediction_cache)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    base_predictions = cache["base"]
    scale_predictions = cache["scale"]
    targets = cache["targets"]

    data_dir = ROOT / "data" / args.split
    dataset = YOLOTinyDataset(
        img_dir=data_dir / "images",
        lbl_dir=data_dir / "labels",
        is_train=False,
    )
    if len(dataset.tile_index) != len(targets):
        raise ValueError(
            "Dataset tile index and cached targets have different lengths: "
            f"{len(dataset.tile_index)} != {len(targets)}"
        )

    tune_indices = [
        index
        for index, tile in enumerate(dataset.tile_index)
        if int(tile[0]) % 2 == 0
    ]
    confirm_indices = [
        index
        for index, tile in enumerate(dataset.tile_index)
        if int(tile[0]) % 2 == 1
    ]
    rules = _candidate_rules()
    predictions = {name: [] for name in rules}
    matched_pairs = 0
    base_detections = 0
    scale_detections = 0
    for base, scale in zip(base_predictions, scale_predictions):
        pairing = _greedy_cross_view_pairs(
            base,
            scale,
            args.pair_threshold,
        )
        matched_pairs += len(pairing[0])
        base_detections += len(base["boxes"])
        scale_detections += len(scale["boxes"])
        for name, rule in rules.items():
            predictions[name].append(
                _adaptive_pair_fuse(
                    base,
                    scale,
                    pairing=pairing,
                    rule=rule,
                    pair_threshold=args.pair_threshold,
                    detections_per_image=args.detections_per_image,
                )
            )

    tune_targets = _subset(targets, tune_indices)
    tune_metrics = {}
    for name, prediction in predictions.items():
        print(f"Tuning {name}...")
        tune_metrics[name] = evaluate_coco(
            _subset(prediction, tune_indices),
            tune_targets,
            class_metrics=False,
        )

    shortlist = {"pair_only", "unmatched_w75"}
    for metric_name in ("coco_AP", "coco_AP75", "coco_AR100"):
        shortlist.update(
            _ranked_names(
                tune_metrics,
                metric_name,
                args.shortlist_per_metric,
            )
        )

    confirm_targets = _subset(targets, confirm_indices)
    confirm_metrics = {}
    full_metrics = {}
    for name in sorted(shortlist):
        print(f"Confirming {name}...")
        confirm_metrics[name] = evaluate_coco(
            _subset(predictions[name], confirm_indices),
            confirm_targets,
            class_metrics=False,
        )
        print(f"Evaluating full validation for {name}...")
        full_metrics[name] = evaluate_coco(
            predictions[name],
            targets,
            class_metrics=False,
        )

    leader_names = {
        metric_name: max(
            shortlist,
            key=lambda name: full_metrics[name][metric_name],
        )
        for metric_name in ("coco_AP", "coco_AP75", "coco_AR100")
    }
    robust_leader_names = {
        metric_name: _robust_leader(
            shortlist,
            tune_metrics,
            confirm_metrics,
            full_metrics,
            metric_name,
        )
        for metric_name in ("coco_AP", "coco_AP75", "coco_AR100")
    }
    detail_names = (
        set(leader_names.values())
        | set(robust_leader_names.values())
        | {"pair_only", "unmatched_w75"}
    )
    leader_details = {
        name: {
            **full_metrics[name],
            **compute_scale_ap(predictions[name], targets),
            **compute_class_aware_scale_ap(predictions[name], targets),
        }
        for name in sorted(detail_names)
    }
    ap75_audit = {
        name: _audit_predictions(predictions[name], targets, dataset)
        for name in sorted(detail_names)
    }

    summary = {
        "prediction_cache": str(cache_path),
        "checkpoint": cache.get("checkpoint"),
        "profile": cache.get("profile"),
        "split": args.split,
        "tiles": len(targets),
        "original_images": len({int(tile[0]) for tile in dataset.tile_index}),
        "folds": {
            "tune_even_original_images": {
                "tiles": len(tune_indices),
                "original_images": len(
                    {int(dataset.tile_index[index][0]) for index in tune_indices}
                ),
            },
            "confirm_odd_original_images": {
                "tiles": len(confirm_indices),
                "original_images": len(
                    {
                        int(dataset.tile_index[index][0])
                        for index in confirm_indices
                    }
                ),
            },
        },
        "pair_stats": {
            "pair_threshold": args.pair_threshold,
            "matched_pairs": matched_pairs,
            "base_detections": base_detections,
            "scale_detections": scale_detections,
            "match_rate_vs_base": round(
                matched_pairs / max(base_detections, 1),
                6,
            ),
        },
        "candidate_rules": rules,
        "tune_metrics": tune_metrics,
        "shortlist": sorted(shortlist),
        "confirm_metrics": confirm_metrics,
        "full_metrics": full_metrics,
        "leader_names": leader_names,
        "robust_leader_names": robust_leader_names,
        "leader_details": leader_details,
        "ap75_audit": ap75_audit,
    }

    if args.save_predictions is not None:
        prediction_path = _resolve(args.save_predictions)
        prediction_path.parent.mkdir(parents=True, exist_ok=True)
        saved_names = sorted(detail_names)
        torch.save(
            {
                "checkpoint": cache.get("checkpoint"),
                "profile": cache.get("profile"),
                "source_prediction_cache": str(cache_path),
                "candidate_rules": {
                    name: rules[name] for name in saved_names
                },
                "predictions": {
                    name: predictions[name] for name in saved_names
                },
                "targets": targets,
            },
            prediction_path,
        )
        summary["saved_predictions"] = str(prediction_path)

    out_path = _resolve(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "shortlist": sorted(shortlist),
                "leader_names": leader_names,
                "robust_leader_names": robust_leader_names,
                "leader_details": leader_details,
                "saved": str(out_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
