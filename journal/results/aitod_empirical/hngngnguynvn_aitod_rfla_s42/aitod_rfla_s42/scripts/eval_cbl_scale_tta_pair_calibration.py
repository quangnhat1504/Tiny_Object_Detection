"""Tune matched-pair coordinate and score calibration for scale TTA."""
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
    _finalize,
    _greedy_cross_view_pairs,
)
from scripts.eval_cbl_scale_tta import _audit_predictions
from scripts.eval_cbl_scale_tta_adaptive_unmatched import (
    _ranked_names,
    _resolve,
    _robust_leader,
    _subset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune matched base/scale box and score calibration on one "
            "original-image fold, then confirm on the other fold"
        )
    )
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument("--split", choices=("valid",), default="valid")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save-predictions", type=Path, default=None)
    parser.add_argument("--pair-threshold", type=float, default=0.50)
    parser.add_argument("--unmatched-scale-weight", type=float, default=0.75)
    parser.add_argument("--detections-per-image", type=int, default=200)
    parser.add_argument("--shortlist-per-metric", type=int, default=3)
    parser.add_argument(
        "--search-profile",
        choices=("broad", "fine", "combined"),
        default="broad",
    )
    return parser.parse_args()


def _candidate_rules(
    search_profile: str = "broad",
) -> dict[str, dict[str, Any]]:
    rules = {
        "current": {
            "coordinate_mode": "score_power",
            "coordinate_power": 1.0,
            "score_mode": "mean",
        },
        "pair_only_current": {
            "coordinate_mode": "score_power",
            "coordinate_power": 1.0,
            "score_mode": "mean",
            "include_unmatched_scale": False,
        },
    }
    if search_profile in ("fine", "combined"):
        size_rules = set()
        if search_profile == "fine":
            for cutoff in (10.0, 12.0, 14.0, 16.0, 18.0):
                size_rules.add((cutoff, 0.75, 0.50))
            for cutoff in (12.0, 16.0):
                for below_alpha in (0.65, 0.85):
                    size_rules.add((cutoff, below_alpha, 0.50))
                for above_alpha in (0.40, 0.60):
                    size_rules.add((cutoff, 0.75, above_alpha))
        else:
            for cutoff in (12.0, 16.0, 18.0):
                for below_alpha in (0.80, 0.85):
                    for above_alpha in (0.40, 0.45):
                        size_rules.add(
                            (cutoff, below_alpha, above_alpha)
                        )
        for cutoff, below_alpha, above_alpha in sorted(size_rules):
            rules[
                f"coord_size_lt{int(cutoff):02d}"
                f"_a{int(below_alpha * 100):02d}"
                f"_else{int(above_alpha * 100):02d}"
            ] = {
                "coordinate_mode": "size_step",
                "size_cutoff": cutoff,
                "scale_alpha_below": below_alpha,
                "scale_alpha_above": above_alpha,
                "score_mode": "mean",
            }
        return rules
    if search_profile != "broad":
        raise ValueError(f"Unknown search profile: {search_profile}")

    for alpha in (0.25, 0.40, 0.50, 0.60, 0.75):
        rules[f"coord_fixed_a{int(alpha * 100):02d}"] = {
            "coordinate_mode": "fixed",
            "scale_alpha": alpha,
            "score_mode": "mean",
        }
    for power in (0.50, 2.00):
        rules[f"coord_score_power{int(power * 100):03d}"] = {
            "coordinate_mode": "score_power",
            "coordinate_power": power,
            "score_mode": "mean",
        }
    for cutoff in (8.0, 12.0, 16.0):
        rules[f"coord_size_lt{int(cutoff):02d}_a75_else50"] = {
            "coordinate_mode": "size_step",
            "size_cutoff": cutoff,
            "scale_alpha_below": 0.75,
            "scale_alpha_above": 0.50,
            "score_mode": "mean",
        }
    for score_mode in ("geometric", "minimum", "maximum"):
        rules[f"score_{score_mode}"] = {
            "coordinate_mode": "score_power",
            "coordinate_power": 1.0,
            "score_mode": score_mode,
        }
    for power in (0.10, 0.25, 0.50):
        rules[f"score_mean_agreement_power{int(power * 100):02d}"] = {
            "coordinate_mode": "score_power",
            "coordinate_power": 1.0,
            "score_mode": "mean",
            "agreement_power": power,
        }
    for power in (0.10, 0.25):
        rules[f"coord_fixed_a50_score_agreement_power{int(power * 100):02d}"] = {
            "coordinate_mode": "fixed",
            "scale_alpha": 0.50,
            "score_mode": "mean",
            "agreement_power": power,
        }
    return rules


def _pair_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    top_left = torch.maximum(first[:2], second[:2])
    bottom_right = torch.minimum(first[2:], second[2:])
    intersection = (bottom_right - top_left).clamp(min=0.0).prod()
    first_area = (first[2:] - first[:2]).clamp(min=0.0).prod()
    second_area = (second[2:] - second[:2]).clamp(min=0.0).prod()
    return intersection / (first_area + second_area - intersection).clamp(
        min=1e-8
    )


def _coordinate_alpha(
    base_box: torch.Tensor,
    base_score: torch.Tensor,
    scale_score: torch.Tensor,
    rule: dict[str, Any],
) -> torch.Tensor:
    mode = rule["coordinate_mode"]
    if mode == "fixed":
        return base_score.new_tensor(float(rule["scale_alpha"]))
    if mode == "size_step":
        width = (base_box[2] - base_box[0]).clamp(min=0.0)
        height = (base_box[3] - base_box[1]).clamp(min=0.0)
        size = (width * height).sqrt()
        alpha = (
            rule["scale_alpha_below"]
            if float(size) < float(rule["size_cutoff"])
            else rule["scale_alpha_above"]
        )
        return base_score.new_tensor(float(alpha))
    raise ValueError(f"Unknown coordinate mode: {mode}")


def _pair_score(
    base_score: torch.Tensor,
    scale_score: torch.Tensor,
    agreement: torch.Tensor,
    rule: dict[str, Any],
) -> torch.Tensor:
    mode = rule["score_mode"]
    if mode == "mean":
        score = (base_score + scale_score) / 2
    elif mode == "geometric":
        score = (base_score * scale_score).clamp(min=0.0).sqrt()
    elif mode == "minimum":
        score = torch.minimum(base_score, scale_score)
    elif mode == "maximum":
        score = torch.maximum(base_score, scale_score)
    else:
        raise ValueError(f"Unknown score mode: {mode}")
    if "agreement_power" in rule:
        score = score * agreement.clamp(min=1e-8).pow(
            float(rule["agreement_power"])
        )
    return score


def _calibrated_pair_fuse(
    base: dict[str, torch.Tensor],
    scale: dict[str, torch.Tensor],
    *,
    pairing: tuple[dict[int, int], set[int]],
    rule: dict[str, Any],
    unmatched_scale_weight: float,
    detections_per_image: int,
) -> dict[str, torch.Tensor]:
    matches, used_scale = pairing
    boxes, scores, labels = [], [], []
    for base_index in range(len(base["boxes"])):
        box = base["boxes"][base_index]
        score = base["scores"][base_index]
        scale_index = matches.get(base_index)
        if scale_index is not None:
            scale_box = scale["boxes"][scale_index]
            scale_score = scale["scores"][scale_index]
            agreement = _pair_iou(box, scale_box)
            if rule["coordinate_mode"] == "score_power":
                power = float(rule["coordinate_power"])
                coordinate_weights = torch.stack(
                    (
                        score.clamp(min=1e-8).pow(power),
                        scale_score.clamp(min=1e-8).pow(power),
                    )
                )
                box = (
                    coordinate_weights[0] * box
                    + coordinate_weights[1] * scale_box
                ) / coordinate_weights.sum()
            else:
                alpha = _coordinate_alpha(
                    box,
                    score,
                    scale_score,
                    rule,
                )
                box = (1.0 - alpha) * box + alpha * scale_box
            score = _pair_score(
                score,
                scale_score,
                agreement,
                rule,
            )
        boxes.append(box)
        scores.append(score)
        labels.append(base["labels"][base_index])

    if rule.get("include_unmatched_scale", True):
        for scale_index in range(len(scale["boxes"])):
            if scale_index in used_scale:
                continue
            boxes.append(scale["boxes"][scale_index])
            scores.append(
                scale["scores"][scale_index] * unmatched_scale_weight
            )
            labels.append(scale["labels"][scale_index])

    if not boxes:
        return {
            "boxes": torch.zeros((0, 4)),
            "scores": torch.zeros((0,)),
            "labels": torch.zeros((0,), dtype=torch.int64),
        }
    return _finalize(
        torch.stack(boxes),
        torch.stack(scores),
        torch.stack(labels),
        nms_threshold=0.50,
        detections_per_image=detections_per_image,
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

    rules = _candidate_rules(args.search_profile)
    predictions = {name: [] for name in rules}
    matched_pairs = 0
    for base, scale in zip(base_predictions, scale_predictions):
        pairing = _greedy_cross_view_pairs(
            base,
            scale,
            args.pair_threshold,
        )
        matched_pairs += len(pairing[0])
        for name, rule in rules.items():
            predictions[name].append(
                _calibrated_pair_fuse(
                    base,
                    scale,
                    pairing=pairing,
                    rule=rule,
                    unmatched_scale_weight=args.unmatched_scale_weight,
                    detections_per_image=args.detections_per_image,
                )
            )

    tune_targets = _subset(targets, tune_indices)
    tune_metrics = {}
    for name, prediction in predictions.items():
        print(f"Tuning {name}...", flush=True)
        tune_metrics[name] = evaluate_coco(
            _subset(prediction, tune_indices),
            tune_targets,
            class_metrics=False,
        )

    shortlist = {"current", "pair_only_current"}
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
        print(f"Confirming {name}...", flush=True)
        confirm_metrics[name] = evaluate_coco(
            _subset(predictions[name], confirm_indices),
            confirm_targets,
            class_metrics=False,
        )
        print(f"Evaluating full validation for {name}...", flush=True)
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
            baseline="current",
        )
        for metric_name in ("coco_AP", "coco_AP75", "coco_AR100")
    }
    detail_names = (
        set(leader_names.values())
        | set(robust_leader_names.values())
        | {"current", "pair_only_current"}
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
            "unmatched_scale_weight": args.unmatched_scale_weight,
        },
        "search_profile": args.search_profile,
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
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
