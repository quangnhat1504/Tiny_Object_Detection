"""Compare RPN proposal quality for the same checkpoint at two scales.

The audit keeps every ground-truth instance aligned across transforms and
measures whether a high-resolution RPN provides complementary proposal recall.
It is validation-only and does not update model weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision.ops import box_iou

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import SEED, seed_all
from common.dataset import YOLOTinyDataset, collate_fn
from scripts.audit_rpn_proposal_recall import (
    SIZE_BINS,
    build_checkpoint_model,
    size_bin_masks,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit cross-scale RPN proposal complementarity")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument(
        "--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--top-n", type=int, nargs="+", default=(100, 300, 1500))
    parser.add_argument(
        "--iou-thresholds", type=float, nargs="+", default=(0.5, 0.75))
    parser.add_argument("--base-min-size", type=int, default=800)
    parser.add_argument("--base-max-size", type=int, default=800)
    parser.add_argument("--teacher-min-size", type=int, default=960)
    parser.add_argument("--teacher-max-size", type=int, default=1200)
    parser.add_argument("--max-tiles", type=int, default=None)
    parser.add_argument(
        "--out",
        default="runs/rpn_cross_scale_complementarity_valid.json",
    )
    return parser.parse_args()


def _rpn_forward(
    model: torch.nn.Module,
    images: list[torch.Tensor],
    targets: list[dict],
    min_size: int,
    max_size: int,
) -> tuple[list[torch.Tensor], list[dict]]:
    model.transform.min_size = (min_size,)
    model.transform.max_size = max_size
    image_list, transformed_targets = model.transform(images, targets)
    features = model.backbone(image_list.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    proposals, _ = model.rpn(image_list, features)
    return proposals, transformed_targets


def _empty_band_stats(
    top_ns: list[int], thresholds: list[float]
) -> dict:
    return {
        f"top{top_n}": {
            "count": 0,
            "teacher_wins": 0,
            "base_wins": 0,
            "ties": 0,
            "delta_sum": 0.0,
            "delta_values": [],
            **{
                f"iou{threshold:g}": {
                    "base_hits": 0,
                    "teacher_hits": 0,
                    "oracle_hits": 0,
                    "teacher_rescues": 0,
                    "teacher_regressions": 0,
                }
                for threshold in thresholds
            },
        }
        for top_n in top_ns
    }


def _max_iou(
    gt_boxes: torch.Tensor,
    proposals: torch.Tensor,
    top_n: int,
) -> torch.Tensor:
    selected = proposals[:top_n]
    if selected.numel() == 0:
        return torch.zeros(len(gt_boxes), device=gt_boxes.device)
    return box_iou(gt_boxes, selected).max(dim=1).values


def _update_stats(
    stats: dict,
    top_n: int,
    thresholds: list[float],
    base_iou: torch.Tensor,
    teacher_iou: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    base = base_iou[mask]
    teacher = teacher_iou[mask]
    if base.numel() == 0:
        return
    delta = teacher - base
    top_stats = stats[f"top{top_n}"]
    top_stats["count"] += int(delta.numel())
    top_stats["teacher_wins"] += int((delta > 1e-12).sum().item())
    top_stats["base_wins"] += int((delta < -1e-12).sum().item())
    top_stats["ties"] += int((delta.abs() <= 1e-12).sum().item())
    top_stats["delta_sum"] += float(delta.sum().item())
    top_stats["delta_values"].append(delta.cpu())
    for threshold in thresholds:
        key = f"iou{threshold:g}"
        base_hit = base >= threshold
        teacher_hit = teacher >= threshold
        threshold_stats = top_stats[key]
        threshold_stats["base_hits"] += int(base_hit.sum().item())
        threshold_stats["teacher_hits"] += int(teacher_hit.sum().item())
        threshold_stats["oracle_hits"] += int(
            (base_hit | teacher_hit).sum().item())
        threshold_stats["teacher_rescues"] += int(
            ((~base_hit) & teacher_hit).sum().item())
        threshold_stats["teacher_regressions"] += int(
            (base_hit & (~teacher_hit)).sum().item())


def _finalize_stats(stats: dict) -> dict:
    result = {}
    for top_key, values in stats.items():
        count = values["count"]
        delta_tensors = values.pop("delta_values")
        deltas = (
            torch.cat(delta_tensors) if delta_tensors else torch.empty(0))
        summary = {
            "count": count,
            "teacher_win_rate": (
                values["teacher_wins"] / count if count else 0.0),
            "base_win_rate": (
                values["base_wins"] / count if count else 0.0),
            "tie_rate": values["ties"] / count if count else 0.0,
            "mean_teacher_minus_base_max_iou": (
                values["delta_sum"] / count if count else 0.0),
            "delta_quantiles": {
                key: float(value)
                for key, value in zip(
                    ("min", "p10", "p25", "median", "p75", "p90", "max"),
                    torch.quantile(
                        deltas,
                        torch.tensor(
                            (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
                            dtype=deltas.dtype,
                        ),
                    ).tolist() if deltas.numel() else (0.0,) * 7,
                )
            },
        }
        for key, threshold_stats in values.items():
            if not key.startswith("iou"):
                continue
            summary[key] = {
                "base_recall": (
                    threshold_stats["base_hits"] / count if count else 0.0),
                "teacher_recall": (
                    threshold_stats["teacher_hits"] / count
                    if count else 0.0),
                "oracle_union_recall": (
                    threshold_stats["oracle_hits"] / count
                    if count else 0.0),
                "teacher_rescues": threshold_stats["teacher_rescues"],
                "teacher_regressions": (
                    threshold_stats["teacher_regressions"]),
            }
        result[top_key] = summary
    return result


def main() -> None:
    args = parse_args()
    seed_all(SEED)
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if min(
        args.base_min_size,
        args.base_max_size,
        args.teacher_min_size,
        args.teacher_max_size,
    ) < 1:
        raise ValueError("All transform sizes must be positive")
    if args.max_tiles is not None and args.max_tiles < 1:
        raise ValueError("--max-tiles must be positive")
    top_ns = sorted(set(args.top_n))
    thresholds = sorted(set(args.iou_thresholds))
    if not top_ns or top_ns[0] < 1:
        raise ValueError("--top-n values must be positive")
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("--iou-thresholds values must be in [0, 1]")

    checkpoint_path = Path(args.ckpt)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    model = build_checkpoint_model(checkpoint, device)

    data_dir = ROOT / "data" / args.split
    dataset = YOLOTinyDataset(
        img_dir=data_dir / "images",
        lbl_dir=data_dir / "labels",
        is_train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    stats = {
        "overall": _empty_band_stats(top_ns, thresholds),
        **{
            name: _empty_band_stats(top_ns, thresholds)
            for name, _, _ in SIZE_BINS
        },
    }
    counts = {key: 0 for key in stats}
    num_tiles = 0

    with torch.inference_mode():
        for images, targets in loader:
            if args.max_tiles is not None and num_tiles >= args.max_tiles:
                break
            remaining = (
                len(images)
                if args.max_tiles is None
                else min(len(images), args.max_tiles - num_tiles)
            )
            images = [
                image.to(device, non_blocking=True)
                for image in images[:remaining]
            ]
            original_targets = [
                {
                    key: (
                        value.to(device, non_blocking=True)
                        if torch.is_tensor(value) else value)
                    for key, value in target.items()
                }
                for target in targets[:remaining]
            ]
            base_proposals, base_targets = _rpn_forward(
                model,
                images,
                original_targets,
                args.base_min_size,
                args.base_max_size,
            )
            teacher_proposals, teacher_targets = _rpn_forward(
                model,
                images,
                original_targets,
                args.teacher_min_size,
                args.teacher_max_size,
            )

            for original, base_target, teacher_target, base_prop, teacher_prop in zip(
                original_targets,
                base_targets,
                teacher_targets,
                base_proposals,
                teacher_proposals,
            ):
                num_tiles += 1
                if len(base_target["boxes"]) != len(teacher_target["boxes"]):
                    raise AssertionError("Cross-scale GT order/count changed")
                num_gt = len(base_target["boxes"])
                if num_gt == 0:
                    continue
                masks = size_bin_masks(original["boxes"])
                counts["overall"] += num_gt
                for name, mask in masks.items():
                    counts[name] += int(mask.sum().item())
                all_mask = torch.ones(
                    num_gt, dtype=torch.bool, device=device)
                for top_n in top_ns:
                    base_iou = _max_iou(
                        base_target["boxes"], base_prop, top_n)
                    teacher_iou = _max_iou(
                        teacher_target["boxes"], teacher_prop, top_n)
                    _update_stats(
                        stats["overall"],
                        top_n,
                        thresholds,
                        base_iou,
                        teacher_iou,
                        all_mask,
                    )
                    for name, mask in masks.items():
                        _update_stats(
                            stats[name],
                            top_n,
                            thresholds,
                            base_iou,
                            teacher_iou,
                            mask,
                        )
            if num_tiles % 200 < args.batch_size:
                total = min(len(dataset), args.max_tiles or len(dataset))
                print(f"Audited {num_tiles}/{total} tiles", flush=True)

    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_model_source": checkpoint.get(
            "model_source", "legacy_unspecified"),
        "split": args.split,
        "num_tiles": num_tiles,
        "gt_counts": counts,
        "base_transform": [args.base_min_size, args.base_max_size],
        "teacher_transform": [
            args.teacher_min_size, args.teacher_max_size],
        "top_n": top_ns,
        "iou_thresholds": thresholds,
        "size_bins_sqrt_area_px": {
            name: [lower, None if upper == float("inf") else upper]
            for name, lower, upper in SIZE_BINS
        },
        "complementarity": {
            name: _finalize_stats(band_stats)
            for name, band_stats in stats.items()
        },
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["complementarity"], indent=2))
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
