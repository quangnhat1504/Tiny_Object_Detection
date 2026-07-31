"""Evaluate fixed, fused, cross-fit, and oracle CBL refinement trajectories."""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from torchvision.ops import boxes as box_ops
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import common.model as model_module
from common.config import SEED, seed_all
from common.dataset import YOLOTinyDataset, collate_fn
from common.eval_utils import evaluate_coco
from scripts.analyze_refinement_consistency import (
    _build_model_from_checkpoint,
    _rescale_boxes,
)


SIZE_EDGES_PX = (8.0, 12.0, 20.0, 32.0)
STABILITY_EDGES = (0.75, 0.90, 0.97)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cache CBL box trajectories, estimate a cross-fit pass selector, "
            "and compare it with fixed passes, trajectory fusion, and a GT oracle."
        )
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", choices=("valid",), default="valid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--score-threshold", type=float, default=None)
    parser.add_argument(
        "--detector-score-threshold",
        type=float,
        default=0.001,
        help="Initial RoI detection threshold; matches common.eval_utils.evaluate",
    )
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _bin_index(value: float, edges: tuple[float, ...]) -> int:
    for index, edge in enumerate(edges):
        if value < edge:
            return index
    return len(edges)


def _bin_name(index: int, edges: tuple[float, ...]) -> str:
    if index == 0:
        return f"<{edges[0]:g}"
    if index == len(edges):
        return f">={edges[-1]:g}"
    return f"{edges[index - 1]:g}-{edges[index]:g}"


def _equivalent_size_px(boxes: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    widths_heights = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
    normalized = widths_heights.prod(dim=1).sqrt() / math.sqrt(
        image_shape[0] * image_shape[1]
    )
    return normalized * 512.0


def _self_iou_rows(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.numel() == 0:
        return first.new_zeros((0,))
    intersection_lt = torch.maximum(first[:, :2], second[:, :2])
    intersection_rb = torch.minimum(first[:, 2:], second[:, 2:])
    intersection = (intersection_rb - intersection_lt).clamp(min=0).prod(dim=1)
    area_first = (first[:, 2:] - first[:, :2]).clamp(min=0).prod(dim=1)
    area_second = (second[:, 2:] - second[:, :2]).clamp(min=0).prod(dim=1)
    union = area_first + area_second - intersection
    return intersection / union.clamp(min=1e-12)


def _same_class_iou_matrix(
    trajectory: list[torch.Tensor],
    labels: torch.Tensor,
    gt_boxes: torch.Tensor,
    gt_labels: torch.Tensor,
) -> torch.Tensor:
    values = trajectory[0].new_zeros((len(labels), len(trajectory)))
    for label in labels.unique().tolist():
        pred_indices = torch.where(labels == int(label))[0]
        gt_indices = torch.where(gt_labels == int(label))[0]
        if pred_indices.numel() == 0 or gt_indices.numel() == 0:
            continue
        for pass_index, boxes in enumerate(trajectory):
            overlaps = box_iou(boxes[pred_indices], gt_boxes[gt_indices])
            values[pred_indices, pass_index] = overlaps.max(dim=1).values
    return values


def _finalize_prediction(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    nms_threshold: float,
    detections_per_image: int,
) -> dict[str, torch.Tensor]:
    finite = torch.isfinite(boxes).all(dim=1)
    boxes, scores, labels = boxes[finite], scores[finite], labels[finite]
    keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
    boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
    keep = box_ops.batched_nms(boxes, scores, labels, nms_threshold)
    keep = keep[:detections_per_image]
    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep],
    }


def _trajectory_refiner(captures: list[dict]):
    def observed_refine(
        roi_heads,
        features,
        boxes,
        scores,
        labels,
        image_shapes,
        steps,
        blend,
        last_step_blend,
        score_threshold,
        extra_min_size_ratio,
    ):
        del last_step_blend, extra_min_size_ratio
        current_boxes = boxes
        trajectory = [[value.detach().cpu() for value in current_boxes]]
        for _ in range(steps):
            if not any(value.numel() for value in current_boxes):
                break
            pooled = roi_heads.box_roi_pool(features, current_boxes, image_shapes)
            box_features = roi_heads.box_head(pooled)
            predictor_out = roi_heads.box_predictor(box_features)
            if (
                not getattr(roi_heads.box_predictor, "is_distributional", False)
                or len(predictor_out) != 3
            ):
                raise RuntimeError(
                    "Trajectory analysis requires a distributional CBL predictor"
                )
            _, box_regression, _ = predictor_out
            decoded = roi_heads.box_coder.decode(box_regression, current_boxes)
            decoded_per_image = decoded.split(
                [len(value) for value in current_boxes], dim=0
            )
            refined_boxes = []
            for (
                decoded_boxes,
                boxes_per_image,
                scores_per_image,
                labels_per_image,
                image_shape,
            ) in zip(
                decoded_per_image,
                current_boxes,
                scores,
                labels,
                image_shapes,
            ):
                if decoded_boxes.numel() == 0:
                    refined_boxes.append(decoded_boxes.reshape(0, 4))
                    continue
                row_ids = torch.arange(
                    len(labels_per_image), device=decoded_boxes.device
                )
                selected = decoded_boxes[row_ids, labels_per_image]
                selected = boxes_per_image + blend * (
                    selected - boxes_per_image
                )
                if score_threshold > 0:
                    selected = torch.where(
                        (scores_per_image >= score_threshold).unsqueeze(1),
                        selected,
                        boxes_per_image,
                    )
                refined_boxes.append(
                    box_ops.clip_boxes_to_image(selected, image_shape)
                )
            current_boxes = refined_boxes
            trajectory.append(
                [value.detach().cpu() for value in current_boxes]
            )

        while len(trajectory) < steps + 1:
            trajectory.append(
                [value.clone() for value in trajectory[-1]]
            )
        captures.append(
            {
                "trajectory": trajectory,
                "scores": [value.detach().cpu() for value in scores],
                "labels": [value.detach().cpu() for value in labels],
                "image_shapes": [tuple(shape) for shape in image_shapes],
            }
        )

        final_boxes, final_scores, final_labels = [], [], []
        for boxes_per_image, scores_per_image, labels_per_image in zip(
            current_boxes, scores, labels
        ):
            prediction = _finalize_prediction(
                boxes_per_image,
                scores_per_image,
                labels_per_image,
                roi_heads.nms_thresh,
                roi_heads.detections_per_img,
            )
            final_boxes.append(prediction["boxes"])
            final_scores.append(prediction["scores"])
            final_labels.append(prediction["labels"])
        return final_boxes, final_scores, final_labels

    return observed_refine


def _collect_tiles(
    model,
    loader: DataLoader,
    device: torch.device,
    steps: int,
) -> list[dict]:
    captures: list[dict] = []
    original_refine = model_module._iteratively_refine_cbl_detections
    model_module._iteratively_refine_cbl_detections = _trajectory_refiner(captures)
    tiles: list[dict] = []
    try:
        for images, targets in tqdm(loader, desc="CBL trajectories"):
            capture_start = len(captures)
            original_shapes = [
                (int(image.shape[-2]), int(image.shape[-1]))
                for image in images
            ]
            with torch.no_grad(), torch.amp.autocast(
                "cuda", enabled=device.type == "cuda"
            ):
                model([image.to(device) for image in images])
            if len(captures) != capture_start + 1:
                raise RuntimeError("Expected exactly one trajectory capture per batch")
            capture = captures.pop()
            for batch_index, (target, original_shape) in enumerate(
                zip(targets, original_shapes)
            ):
                transformed_shape = capture["image_shapes"][batch_index]
                trajectory = [
                    _rescale_boxes(
                        capture["trajectory"][pass_index][batch_index],
                        transformed_shape,
                        original_shape,
                    )
                    for pass_index in range(steps + 1)
                ]
                labels = capture["labels"][batch_index]
                gt_boxes = target["boxes"].cpu()
                gt_labels = target["labels"].cpu()
                tiles.append(
                    {
                        "trajectory": trajectory,
                        "scores": capture["scores"][batch_index],
                        "labels": labels,
                        "image_shape": original_shape,
                        "gt": {
                            key: value.cpu() if isinstance(value, torch.Tensor) else value
                            for key, value in target.items()
                        },
                        "same_class_ious": _same_class_iou_matrix(
                            trajectory, labels, gt_boxes, gt_labels
                        ),
                    }
                )
    finally:
        model_module._iteratively_refine_cbl_detections = original_refine
    return tiles


def _feature_key(tile: dict, index: int, include_stability: bool) -> tuple[int, ...]:
    size = float(
        _equivalent_size_px(
            tile["trajectory"][0][index : index + 1],
            tile["image_shape"],
        )[0]
    )
    size_bin = _bin_index(size, SIZE_EDGES_PX)
    if not include_stability:
        return (size_bin,)
    stability = float(
        _self_iou_rows(
            tile["trajectory"][0][index : index + 1],
            tile["trajectory"][1][index : index + 1],
        )[0]
    )
    return size_bin, _bin_index(stability, STABILITY_EDGES)


def _fit_rule(
    tiles: list[dict],
    train_fold: int,
    score_threshold: float,
    include_stability: bool,
) -> dict:
    stats: dict[tuple[int, ...], dict] = defaultdict(
        lambda: {
            "n": 0,
            "iou_sum": [0.0] * len(tiles[0]["trajectory"]),
            "iou75": [0] * len(tiles[0]["trajectory"]),
        }
    )
    global_stats = {
        "n": 0,
        "iou_sum": [0.0] * len(tiles[0]["trajectory"]),
        "iou75": [0] * len(tiles[0]["trajectory"]),
    }
    for tile_index, tile in enumerate(tiles):
        if tile_index % 2 != train_fold:
            continue
        active = torch.where(tile["scores"] >= score_threshold)[0]
        for index_tensor in active:
            index = int(index_tensor)
            key = _feature_key(tile, index, include_stability)
            ious = tile["same_class_ious"][index].tolist()
            for target in (stats[key], global_stats):
                target["n"] += 1
                for pass_index, iou in enumerate(ious):
                    target["iou_sum"][pass_index] += float(iou)
                    target["iou75"][pass_index] += int(iou >= 0.75)

    def choose_pass(value: dict) -> int:
        return max(
            range(len(value["iou_sum"])),
            key=lambda pass_index: (
                value["iou75"][pass_index] / max(value["n"], 1),
                value["iou_sum"][pass_index] / max(value["n"], 1),
                -pass_index,
            ),
        )

    fallback = choose_pass(global_stats)
    return {
        "passes": {key: choose_pass(value) for key, value in stats.items()},
        "stats": stats,
        "fallback": fallback,
        "train_fold": train_fold,
    }


def _serialize_rule(rule: dict, include_stability: bool) -> dict:
    bins = {}
    for key, pass_index in sorted(rule["passes"].items()):
        size_name = _bin_name(key[0], SIZE_EDGES_PX)
        name = f"size={size_name}px"
        if include_stability:
            name += f",self_iou={_bin_name(key[1], STABILITY_EDGES)}"
        stats = rule["stats"][key]
        count = max(stats["n"], 1)
        bins[name] = {
            "n": stats["n"],
            "selected_pass": pass_index,
            "iou75_rates": [
                round(value / count, 6) for value in stats["iou75"]
            ],
            "mean_ious": [
                round(value / count, 6) for value in stats["iou_sum"]
            ],
        }
    return {
        "train_fold": rule["train_fold"],
        "fallback_pass": rule["fallback"],
        "bins": bins,
    }


def _select_crossfit(
    tile: dict,
    tile_index: int,
    rules: dict[int, dict],
    score_threshold: float,
    include_stability: bool,
) -> torch.Tensor:
    rule = rules[1 - (tile_index % 2)]
    selected = tile["trajectory"][0].clone()
    active = torch.where(tile["scores"] >= score_threshold)[0]
    for index_tensor in active:
        index = int(index_tensor)
        key = _feature_key(tile, index, include_stability)
        pass_index = rule["passes"].get(key, rule["fallback"])
        selected[index] = tile["trajectory"][pass_index][index]
    return selected


def _select_scale12(tile: dict, score_threshold: float) -> torch.Tensor:
    selected = tile["trajectory"][1].clone()
    active = tile["scores"] >= score_threshold
    current = selected
    for pass_index in (2, 3):
        sizes = _equivalent_size_px(current, tile["image_shape"])
        update = active & (sizes >= 12.0)
        current = torch.where(
            update.unsqueeze(1), tile["trajectory"][pass_index], current
        )
    return current


def _select_oracle(tile: dict, score_threshold: float) -> torch.Tensor:
    selected = tile["trajectory"][0].clone()
    best_pass = tile["same_class_ious"].argmax(dim=1)
    active = torch.where(tile["scores"] >= score_threshold)[0]
    for index_tensor in active:
        index = int(index_tensor)
        selected[index] = tile["trajectory"][int(best_pass[index])][index]
    return selected


def _evaluate_selector(
    tiles: list[dict],
    selector: Callable[[dict, int], torch.Tensor],
    nms_threshold: float,
    detections_per_image: int,
    fold: int | None = None,
) -> dict:
    predictions, targets = [], []
    for tile_index, tile in enumerate(tiles):
        if fold is not None and tile_index % 2 != fold:
            continue
        predictions.append(
            _finalize_prediction(
                selector(tile, tile_index),
                tile["scores"],
                tile["labels"],
                nms_threshold,
                detections_per_image,
            )
        )
        targets.append(tile["gt"])
    return evaluate_coco(predictions, targets, class_metrics=False)


def _oracle_summary(tiles: list[dict], score_threshold: float) -> dict:
    counts = [0] * len(tiles[0]["trajectory"])
    total = 0
    p3_iou = 0.0
    oracle_iou = 0.0
    p3_iou75 = 0
    oracle_iou75 = 0
    by_size: dict[int, list[int]] = defaultdict(
        lambda: [0] * len(tiles[0]["trajectory"])
    )
    for tile in tiles:
        active = torch.where(tile["scores"] >= score_threshold)[0]
        sizes = _equivalent_size_px(
            tile["trajectory"][0], tile["image_shape"]
        )
        for index_tensor in active:
            index = int(index_tensor)
            ious = tile["same_class_ious"][index]
            best_pass = int(ious.argmax())
            counts[best_pass] += 1
            by_size[_bin_index(float(sizes[index]), SIZE_EDGES_PX)][
                best_pass
            ] += 1
            total += 1
            p3_iou += float(ious[-1])
            oracle_iou += float(ious[best_pass])
            p3_iou75 += int(float(ious[-1]) >= 0.75)
            oracle_iou75 += int(float(ious[best_pass]) >= 0.75)
    return {
        "n_active": total,
        "best_pass_counts": counts,
        "best_pass_rates": [round(value / max(total, 1), 6) for value in counts],
        "mean_iou_pass3": round(p3_iou / max(total, 1), 6),
        "mean_iou_oracle": round(oracle_iou / max(total, 1), 6),
        "iou75_rate_pass3": round(p3_iou75 / max(total, 1), 6),
        "iou75_rate_oracle": round(oracle_iou75 / max(total, 1), 6),
        "best_pass_counts_by_base_size": {
            _bin_name(key, SIZE_EDGES_PX): value
            for key, value in sorted(by_size.items())
        },
    }


def main() -> None:
    args = parse_args()
    if args.steps != 3:
        raise ValueError("This bounded diagnostic currently requires --steps 3")
    seed_all(SEED)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )
    checkpoint_path = (
        args.ckpt if args.ckpt.is_absolute() else ROOT / args.ckpt
    )
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    model, config = _build_model_from_checkpoint(checkpoint, device)
    score_threshold = (
        float(config.get("cbl_refine_score_threshold", 0.0))
        if args.score_threshold is None
        else args.score_threshold
    )
    model.roi_heads._cbl_refine_steps = args.steps
    model.roi_heads._cbl_refine_score_threshold = score_threshold
    model.roi_heads._cbl_refine_extra_min_size_ratio = 0.0
    model.roi_heads.score_thresh = args.detector_score_threshold

    data_dir = ROOT / "data" / args.split
    dataset = YOLOTinyDataset(
        img_dir=data_dir / "images",
        lbl_dir=data_dir / "labels",
        is_train=False,
    )
    if args.max_tiles is not None:
        dataset = torch.utils.data.Subset(
            dataset, range(min(args.max_tiles, len(dataset)))
        )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )
    tiles = _collect_tiles(model, loader, device, args.steps)
    if len(tiles) < 2:
        raise RuntimeError("Cross-fit trajectory analysis requires at least two tiles")

    size_rules = {
        fold: _fit_rule(tiles, fold, score_threshold, include_stability=False)
        for fold in (0, 1)
    }
    joint_rules = {
        fold: _fit_rule(tiles, fold, score_threshold, include_stability=True)
        for fold in (0, 1)
    }
    nms_threshold = float(model.roi_heads.nms_thresh)
    detections_per_image = int(model.roi_heads.detections_per_img)

    selectors: dict[str, Callable[[dict, int], torch.Tensor]] = {
        **{
            f"pass{pass_index}": (
                lambda tile, _tile_index, value=pass_index: tile["trajectory"][value]
            )
            for pass_index in range(args.steps + 1)
        },
        "scale12_dynamic": (
            lambda tile, _tile_index: _select_scale12(tile, score_threshold)
        ),
        "last_update_025": (
            lambda tile, _tile_index: tile["trajectory"][2]
            + 0.25 * (tile["trajectory"][3] - tile["trajectory"][2])
        ),
        "last_update_050": (
            lambda tile, _tile_index: tile["trajectory"][2]
            + 0.50 * (tile["trajectory"][3] - tile["trajectory"][2])
        ),
        "last_update_075": (
            lambda tile, _tile_index: tile["trajectory"][2]
            + 0.75 * (tile["trajectory"][3] - tile["trajectory"][2])
        ),
        "trajectory_median_123": (
            lambda tile, _tile_index: torch.stack(
                tile["trajectory"][1:4], dim=0
            ).median(dim=0).values
        ),
        "crossfit_size": (
            lambda tile, tile_index: _select_crossfit(
                tile,
                tile_index,
                size_rules,
                score_threshold,
                include_stability=False,
            )
        ),
        "crossfit_size_stability": (
            lambda tile, tile_index: _select_crossfit(
                tile,
                tile_index,
                joint_rules,
                score_threshold,
                include_stability=True,
            )
        ),
        "oracle_gt_pass": (
            lambda tile, _tile_index: _select_oracle(tile, score_threshold)
        ),
    }
    metrics = {}
    for name, selector in selectors.items():
        print(f"Evaluating {name}...")
        metrics[name] = {
            "full": _evaluate_selector(
                tiles,
                selector,
                nms_threshold,
                detections_per_image,
            )
        }
        if name.startswith("crossfit_"):
            metrics[name]["fold0"] = _evaluate_selector(
                tiles,
                selector,
                nms_threshold,
                detections_per_image,
                fold=0,
            )
            metrics[name]["fold1"] = _evaluate_selector(
                tiles,
                selector,
                nms_threshold,
                detections_per_image,
                fold=1,
            )

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_model_source": checkpoint.get(
            "model_source", "legacy_unspecified"
        ),
        "split": args.split,
        "tiles": len(tiles),
        "steps": args.steps,
        "score_threshold": score_threshold,
        "detector_score_threshold": args.detector_score_threshold,
        "size_edges_equivalent_px": SIZE_EDGES_PX,
        "stability_edges": STABILITY_EDGES,
        "oracle_diagnostic": _oracle_summary(tiles, score_threshold),
        "crossfit_size_rules": {
            str(fold): _serialize_rule(rule, include_stability=False)
            for fold, rule in size_rules.items()
        },
        "crossfit_size_stability_rules": {
            str(fold): _serialize_rule(rule, include_stability=True)
            for fold, rule in joint_rules.items()
        },
        "metrics": metrics,
    }
    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
