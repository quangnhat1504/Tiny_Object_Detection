"""Evaluate a detector checkpoint on even/odd original-image validation folds."""
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

from common.dataset import YOLOTinyDataset, collate_fn
from common.eval_utils import (
    compute_class_aware_scale_ap,
    compute_precision_recall,
    compute_scale_ap,
    evaluate_coco,
)
from scripts.analyze_refinement_consistency import _build_model_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def _subset(items: list[dict], indices: list[int]) -> list[dict]:
    return [items[index] for index in indices]


def main() -> None:
    args = parse_args()
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        args.ckpt, map_location="cpu", weights_only=False)
    model, _ = _build_model_from_checkpoint(checkpoint, device)
    model.eval()
    previous_threshold = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = 0.001

    dataset = YOLOTinyDataset(
        img_dir=ROOT / "data/valid/images",
        lbl_dir=ROOT / "data/valid/labels",
        is_train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    predictions: list[dict] = []
    targets_all: list[dict] = []
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Fold inference"):
            with torch.amp.autocast(
                "cuda", enabled=(device.type == "cuda")
            ):
                outputs = model([image.to(device) for image in images])
            predictions.extend(
                {key: value.cpu() for key, value in output.items()}
                for output in outputs
            )
            targets_all.extend(
                {
                    key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in target.items()
                }
                for target in targets
            )
    model.roi_heads.score_thresh = previous_threshold
    assert len(predictions) == len(dataset) == len(targets_all)

    full_scale = compute_scale_ap(predictions, targets_all)
    full_class_aware_scale = compute_class_aware_scale_ap(
        predictions, targets_all)
    total_gt = sum(
        full_scale.get(f"n_gt_{band}", 0)
        for band in ("micro", "tiny", "small", "large")
    )
    primary_scale_ap = (
        sum(
            full_scale.get(f"AP_{band}", 0.0)
            * full_scale.get(f"n_gt_{band}", 0)
            for band in ("micro", "tiny", "small", "large")
        ) / total_gt
        if total_gt
        else 0.0
    )
    full_validation = {
        **evaluate_coco(predictions, targets_all, class_metrics=True),
        **full_scale,
        **full_class_aware_scale,
        **compute_precision_recall(
            predictions, targets_all, iou_thresh=0.5, score_thresh=0.05),
        "mAP_primary": round(primary_scale_ap, 6),
    }

    folds = {}
    for parity, name in ((0, "even_original_images"), (1, "odd_original_images")):
        indices = [
            index
            for index, tile in enumerate(dataset.tile_index)
            if int(tile[0]) % 2 == parity
        ]
        fold_predictions = _subset(predictions, indices)
        fold_targets = _subset(targets_all, indices)
        folds[name] = {
            "tiles": len(indices),
            "original_images": len(
                {int(dataset.tile_index[index][0]) for index in indices}
            ),
            **evaluate_coco(
                fold_predictions, fold_targets, class_metrics=False),
            **compute_scale_ap(fold_predictions, fold_targets),
            **compute_class_aware_scale_ap(
                fold_predictions, fold_targets),
        }

    result = {
        "checkpoint": str(args.ckpt),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_model_source": checkpoint.get("model_source"),
        "tiles": len(dataset),
        "original_images": len(
            {int(tile[0]) for tile in dataset.tile_index}),
        "full_validation": full_validation,
        "folds": folds,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
