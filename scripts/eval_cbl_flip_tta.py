"""Evaluate horizontal-flip TTA and paired box fusion for a CBL checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou
from torchvision.ops import boxes as box_ops
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import SEED, seed_all
from common.dataset import YOLOTinyDataset, collate_fn
from common.eval_utils import (
    compute_class_aware_scale_ap,
    compute_scale_ap,
    evaluate_coco,
)
from scripts.analyze_refinement_consistency import (
    _build_model_from_checkpoint,
)
from scripts.analyze_ap75_errors import analyze_matches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate original/flip predictions and paired TTA fusion"
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", choices=("valid",), default="valid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--cbl-refine-steps", type=int, default=3)
    parser.add_argument("--cbl-refine-blend", type=float, default=1.0)
    parser.add_argument("--cbl-refine-last-step-blend", type=float, default=0.5)
    parser.add_argument("--cbl-refine-last-center-blend", type=float, default=None)
    parser.add_argument("--cbl-refine-last-size-blend", type=float, default=None)
    parser.add_argument("--cbl-refine-score-threshold", type=float, default=0.3)
    parser.add_argument(
        "--cbl-refine-extra-min-size-ratio", type=float, default=0.0
    )
    parser.add_argument("--detector-score-threshold", type=float, default=0.001)
    parser.add_argument(
        "--selected-only",
        action="store_true",
        help="Evaluate only original, flip, and the selected IoU-0.50 fusion",
    )
    parser.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help="Optional torch cache for original, flip, selected, and targets",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _unflip_prediction(
    prediction: dict[str, torch.Tensor],
    image_width: int,
) -> dict[str, torch.Tensor]:
    boxes = prediction["boxes"].detach().cpu().clone()
    old_x1 = boxes[:, 0].clone()
    old_x2 = boxes[:, 2].clone()
    boxes[:, 0] = image_width - old_x2
    boxes[:, 2] = image_width - old_x1
    return {
        "boxes": boxes,
        "scores": prediction["scores"].detach().cpu(),
        "labels": prediction["labels"].detach().cpu(),
    }


def _cpu_prediction(
    prediction: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu()
        for key, value in prediction.items()
        if key in ("boxes", "scores", "labels")
    }


def _finalize(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    *,
    nms_threshold: float = 0.5,
    detections_per_image: int = 200,
) -> dict[str, torch.Tensor]:
    finite = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores)
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


def _union_nms(
    original: dict[str, torch.Tensor],
    flipped: dict[str, torch.Tensor],
    threshold: float,
    detections_per_image: int,
) -> dict[str, torch.Tensor]:
    return _finalize(
        torch.cat((original["boxes"], flipped["boxes"])),
        torch.cat((original["scores"], flipped["scores"])),
        torch.cat((original["labels"], flipped["labels"])),
        nms_threshold=threshold,
        detections_per_image=detections_per_image,
    )


def _greedy_cross_view_pairs(
    original: dict[str, torch.Tensor],
    flipped: dict[str, torch.Tensor],
    threshold: float,
) -> tuple[dict[int, int], set[int]]:
    candidates: list[tuple[float, int, int]] = []
    for label in original["labels"].unique().tolist():
        original_indices = torch.where(original["labels"] == int(label))[0]
        flipped_indices = torch.where(flipped["labels"] == int(label))[0]
        if original_indices.numel() == 0 or flipped_indices.numel() == 0:
            continue
        overlaps = box_iou(
            original["boxes"][original_indices],
            flipped["boxes"][flipped_indices],
        )
        rows, columns = torch.where(overlaps >= threshold)
        candidates.extend(
            (
                float(overlaps[row, column]),
                int(original_indices[row]),
                int(flipped_indices[column]),
            )
            for row, column in zip(rows.tolist(), columns.tolist())
        )
    matches: dict[int, int] = {}
    used_flipped: set[int] = set()
    for _, original_index, flipped_index in sorted(
        candidates, reverse=True
    ):
        if original_index in matches or flipped_index in used_flipped:
            continue
        matches[original_index] = flipped_index
        used_flipped.add(flipped_index)
    return matches, used_flipped


def _pair_fuse(
    original: dict[str, torch.Tensor],
    flipped: dict[str, torch.Tensor],
    *,
    pair_threshold: float,
    coordinate_mode: str,
    score_mode: str,
    include_unmatched_flip: bool,
    unmatched_flip_weight: float,
    detections_per_image: int,
    pairing: tuple[dict[int, int], set[int]] | None = None,
) -> dict[str, torch.Tensor]:
    matches, used_flipped = (
        _greedy_cross_view_pairs(original, flipped, pair_threshold)
        if pairing is None
        else pairing
    )
    boxes, scores, labels = [], [], []
    for original_index in range(len(original["boxes"])):
        box = original["boxes"][original_index]
        score = original["scores"][original_index]
        flipped_index = matches.get(original_index)
        if flipped_index is not None:
            flipped_box = flipped["boxes"][flipped_index]
            flipped_score = flipped["scores"][flipped_index]
            if coordinate_mode == "score_weighted":
                weights = torch.stack((score, flipped_score)).clamp(min=1e-8)
                box = (
                    weights[0] * box + weights[1] * flipped_box
                ) / weights.sum()
            elif coordinate_mode == "equal":
                box = (box + flipped_box) / 2
            else:
                raise ValueError(f"Unknown coordinate mode: {coordinate_mode}")
            if score_mode == "max":
                score = torch.maximum(score, flipped_score)
            elif score_mode == "mean":
                score = (score + flipped_score) / 2
            else:
                raise ValueError(f"Unknown score mode: {score_mode}")
        boxes.append(box)
        scores.append(score)
        labels.append(original["labels"][original_index])

    if include_unmatched_flip:
        for flipped_index in range(len(flipped["boxes"])):
            if flipped_index in used_flipped:
                continue
            boxes.append(flipped["boxes"][flipped_index])
            scores.append(
                flipped["scores"][flipped_index] * unmatched_flip_weight
            )
            labels.append(flipped["labels"][flipped_index])

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
        nms_threshold=0.5,
        detections_per_image=detections_per_image,
    )


def _merge_predictions(
    originals: list[dict[str, torch.Tensor]],
    flipped: list[dict[str, torch.Tensor]],
    detections_per_image: int,
    selected_only: bool,
) -> dict[str, list[dict[str, torch.Tensor]]]:
    merged = {
        "original": originals,
        "flip": flipped,
        "union_nms050": [],
        "union_nms060": [],
    }
    if selected_only:
        merged = {
            "original": originals,
            "flip": flipped,
            "pair_score_weighted_mean_iou50": [],
        }
        for original, flip in zip(originals, flipped):
            pairing = _greedy_cross_view_pairs(original, flip, 0.50)
            merged["pair_score_weighted_mean_iou50"].append(
                _pair_fuse(
                    original,
                    flip,
                    pair_threshold=0.50,
                    coordinate_mode="score_weighted",
                    score_mode="mean",
                    include_unmatched_flip=False,
                    unmatched_flip_weight=1.0,
                    detections_per_image=detections_per_image,
                    pairing=pairing,
                )
            )
        return merged

    for threshold in (0.50, 0.60, 0.70):
        for coordinate_mode in ("equal", "score_weighted"):
            for score_mode in ("max", "mean"):
                name = (
                    f"pair_{coordinate_mode}_{score_mode}_"
                    f"iou{int(threshold * 100):02d}"
                )
                merged[name] = []
    merged["pair_score_weighted_max_iou060_plus_flip090"] = []

    for original, flip in zip(originals, flipped):
        pairings = {
            threshold: _greedy_cross_view_pairs(
                original, flip, threshold
            )
            for threshold in (0.50, 0.60, 0.70)
        }
        merged["union_nms050"].append(
            _union_nms(original, flip, 0.50, detections_per_image)
        )
        merged["union_nms060"].append(
            _union_nms(original, flip, 0.60, detections_per_image)
        )
        for threshold in (0.50, 0.60, 0.70):
            for coordinate_mode in ("equal", "score_weighted"):
                for score_mode in ("max", "mean"):
                    name = (
                        f"pair_{coordinate_mode}_{score_mode}_"
                        f"iou{int(threshold * 100):02d}"
                    )
                    merged[name].append(
                        _pair_fuse(
                            original,
                            flip,
                            pair_threshold=threshold,
                            coordinate_mode=coordinate_mode,
                            score_mode=score_mode,
                            include_unmatched_flip=False,
                            unmatched_flip_weight=1.0,
                            detections_per_image=detections_per_image,
                            pairing=pairings[threshold],
                        )
                    )
        merged["pair_score_weighted_max_iou060_plus_flip090"].append(
            _pair_fuse(
                original,
                flip,
                pair_threshold=0.60,
                coordinate_mode="score_weighted",
                score_mode="max",
                include_unmatched_flip=True,
                unmatched_flip_weight=0.90,
                detections_per_image=detections_per_image,
                pairing=pairings[0.60],
            )
        )
    return merged


def _audit_predictions(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    dataset: YOLOTinyDataset,
    score_threshold: float = 0.05,
    topk: int = 100,
) -> dict:
    filtered = []
    for prediction in predictions:
        keep = torch.where(prediction["scores"] >= score_threshold)[0]
        if keep.numel() > topk:
            order = prediction["scores"][keep].argsort(descending=True)
            keep = keep[order[:topk]]
        filtered.append(
            {key: value[keep] for key, value in prediction.items()}
        )
    summary, _, _, _ = analyze_matches(
        filtered, targets, dataset, edge_margin=4.0
    )
    return {
        "score_threshold": score_threshold,
        "topk": topk,
        **summary,
    }


def main() -> None:
    args = parse_args()
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
    model, _ = _build_model_from_checkpoint(checkpoint, device)
    last_center_blend = (
        args.cbl_refine_last_step_blend
        if args.cbl_refine_last_center_blend is None
        else args.cbl_refine_last_center_blend
    )
    last_size_blend = (
        args.cbl_refine_last_step_blend
        if args.cbl_refine_last_size_blend is None
        else args.cbl_refine_last_size_blend
    )
    model.roi_heads._cbl_refine_steps = args.cbl_refine_steps
    model.roi_heads._cbl_refine_blend = args.cbl_refine_blend
    model.roi_heads._cbl_refine_last_step_blend = (
        args.cbl_refine_last_step_blend
    )
    model.roi_heads._cbl_refine_last_center_blend = last_center_blend
    model.roi_heads._cbl_refine_last_size_blend = last_size_blend
    model.roi_heads._cbl_refine_score_threshold = (
        args.cbl_refine_score_threshold
    )
    model.roi_heads._cbl_refine_extra_min_size_ratio = (
        args.cbl_refine_extra_min_size_ratio
    )
    model.roi_heads.score_thresh = args.detector_score_threshold
    model.eval()

    data_dir = ROOT / "data" / args.split
    base_dataset = YOLOTinyDataset(
        img_dir=data_dir / "images",
        lbl_dir=data_dir / "labels",
        is_train=False,
    )
    dataset = base_dataset
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

    originals, flipped_predictions, targets = [], [], []
    for images, batch_targets in tqdm(loader, desc="Original + flip TTA"):
        device_images = [image.to(device) for image in images]
        flipped_images = [
            torch.flip(image, dims=(-1,)) for image in device_images
        ]
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=device.type == "cuda"
        ):
            original_batch = model(device_images)
            flipped_batch = model(flipped_images)
        for image, original, flipped, target in zip(
            images, original_batch, flipped_batch, batch_targets
        ):
            originals.append(_cpu_prediction(original))
            flipped_predictions.append(
                _unflip_prediction(flipped, int(image.shape[-1]))
            )
            targets.append(
                {
                    key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in target.items()
                }
            )

    all_predictions = _merge_predictions(
        originals,
        flipped_predictions,
        int(model.roi_heads.detections_per_img),
        args.selected_only,
    )
    metrics = {}
    for name, predictions in all_predictions.items():
        print(f"Evaluating {name}...")
        metrics[name] = evaluate_coco(
            predictions, targets, class_metrics=False
        )

    leader_names = {
        max(metrics, key=lambda name: metrics[name]["coco_AP"]),
        max(metrics, key=lambda name: metrics[name]["coco_AP75"]),
        max(metrics, key=lambda name: metrics[name]["coco_AR100"]),
    }
    leader_details = {}
    for name in sorted(leader_names):
        predictions = all_predictions[name]
        leader_details[name] = {
            **compute_scale_ap(predictions, targets),
            **compute_class_aware_scale_ap(predictions, targets),
        }

    summary = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_model_source": checkpoint.get(
            "model_source", "legacy_unspecified"
        ),
        "split": args.split,
        "tiles": len(targets),
        "profile": {
            "cbl_refine_steps": args.cbl_refine_steps,
            "cbl_refine_blend": args.cbl_refine_blend,
            "cbl_refine_last_step_blend": args.cbl_refine_last_step_blend,
            "cbl_refine_last_center_blend": last_center_blend,
            "cbl_refine_last_size_blend": last_size_blend,
            "cbl_refine_score_threshold": (
                args.cbl_refine_score_threshold
            ),
            "cbl_refine_extra_min_size_ratio": (
                args.cbl_refine_extra_min_size_ratio
            ),
            "detector_score_threshold": args.detector_score_threshold,
        },
        "metrics": metrics,
        "leader_details": leader_details,
    }
    selected_name = "pair_score_weighted_mean_iou50"
    if args.max_tiles is None:
        summary["ap75_audit"] = {
            "original": _audit_predictions(
                originals, targets, base_dataset
            ),
            "selected": _audit_predictions(
                all_predictions[selected_name], targets, base_dataset
            ),
        }
    if args.save_predictions is not None:
        cache_path = (
            args.save_predictions
            if args.save_predictions.is_absolute()
            else ROOT / args.save_predictions
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint": str(checkpoint_path),
                "profile": summary["profile"],
                "original": originals,
                "flip": flipped_predictions,
                "selected": all_predictions[selected_name],
                "targets": targets,
            },
            cache_path,
        )
        summary["prediction_cache"] = str(cache_path)
    out_path = args.out if args.out.is_absolute() else ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
