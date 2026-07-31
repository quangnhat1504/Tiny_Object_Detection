"""Evaluate transform-scale TTA for a CBL checkpoint."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import MAX_SIZE, MIN_SIZE, SEED, seed_all
from common.dataset import YOLOTinyDataset, collate_fn
from common.eval_utils import (
    compute_class_aware_scale_ap,
    compute_scale_ap,
    evaluate_coco,
)
from scripts.analyze_ap75_errors import analyze_matches
from scripts.analyze_refinement_consistency import _build_model_from_checkpoint
from scripts.eval_cbl_flip_tta import _cpu_prediction, _pair_fuse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate default-transform and larger-transform scale TTA"
    )
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", choices=("valid",), default="valid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument("--base-min-size", type=int, default=MIN_SIZE)
    parser.add_argument("--base-max-size", type=int, default=MAX_SIZE)
    parser.add_argument("--tta-min-size", type=int, default=800)
    parser.add_argument("--tta-max-size", type=int, default=1000)
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
    parser.add_argument("--pair-threshold", type=float, default=0.50)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--save-predictions",
        type=Path,
        default=None,
        help="Optional torch cache for base, scale, fused, and targets",
    )
    return parser.parse_args()


def _set_transform_size(model: torch.nn.Module, min_size: int, max_size: int) -> None:
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size


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
    return {"score_threshold": score_threshold, "topk": topk, **summary}


def _evaluate_details(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
) -> dict:
    return {
        **compute_scale_ap(predictions, targets),
        **compute_class_aware_scale_ap(predictions, targets),
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

    base_predictions, scale_predictions, fused_predictions, targets = [], [], [], []
    detections_per_image = int(model.roi_heads.detections_per_img)
    for images, batch_targets in tqdm(loader, desc="Base + scale TTA"):
        device_images = [image.to(device) for image in images]
        with torch.no_grad(), torch.amp.autocast(
            "cuda", enabled=device.type == "cuda"
        ):
            _set_transform_size(model, args.base_min_size, args.base_max_size)
            base_batch = model(device_images)
            _set_transform_size(model, args.tta_min_size, args.tta_max_size)
            scale_batch = model(device_images)
        for base, scale, target in zip(base_batch, scale_batch, batch_targets):
            base_prediction = _cpu_prediction(base)
            scale_prediction = _cpu_prediction(scale)
            base_predictions.append(base_prediction)
            scale_predictions.append(scale_prediction)
            fused_predictions.append(
                _pair_fuse(
                    base_prediction,
                    scale_prediction,
                    pair_threshold=args.pair_threshold,
                    coordinate_mode="score_weighted",
                    score_mode="mean",
                    include_unmatched_flip=False,
                    unmatched_flip_weight=1.0,
                    detections_per_image=detections_per_image,
                )
            )
            targets.append(
                {
                    key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in target.items()
                }
            )
    _set_transform_size(model, args.base_min_size, args.base_max_size)

    all_predictions = {
        "base": base_predictions,
        f"scale_min{args.tta_min_size}": scale_predictions,
        f"pair_scale_min{args.tta_min_size}_iou{int(args.pair_threshold * 100):02d}": (
            fused_predictions
        ),
    }
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
    leader_details = {
        name: _evaluate_details(all_predictions[name], targets)
        for name in sorted(leader_names)
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
            "base_min_size": args.base_min_size,
            "base_max_size": args.base_max_size,
            "tta_min_size": args.tta_min_size,
            "tta_max_size": args.tta_max_size,
            "pair_threshold": args.pair_threshold,
            "cbl_refine_steps": args.cbl_refine_steps,
            "cbl_refine_blend": args.cbl_refine_blend,
            "cbl_refine_last_step_blend": args.cbl_refine_last_step_blend,
            "cbl_refine_last_center_blend": last_center_blend,
            "cbl_refine_last_size_blend": last_size_blend,
            "cbl_refine_score_threshold": args.cbl_refine_score_threshold,
            "cbl_refine_extra_min_size_ratio": (
                args.cbl_refine_extra_min_size_ratio
            ),
            "detector_score_threshold": args.detector_score_threshold,
        },
        "metrics": metrics,
        "leader_details": leader_details,
    }
    if args.max_tiles is None:
        fused_name = next(name for name in all_predictions if name.startswith("pair_"))
        summary["ap75_audit"] = {
            "base": _audit_predictions(
                base_predictions, targets, base_dataset
            ),
            f"scale_min{args.tta_min_size}": _audit_predictions(
                scale_predictions, targets, base_dataset
            ),
            fused_name: _audit_predictions(
                fused_predictions, targets, base_dataset
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
                "base": base_predictions,
                "scale": scale_predictions,
                "fused": fused_predictions,
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
