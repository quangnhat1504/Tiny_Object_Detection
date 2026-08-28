"""Measure RPN proposal coverage before RoI classification/regression.

The audit works in transformed image coordinates, where RPN proposals live,
while assigning each ground-truth box to a size bin in the original tile
coordinates. It is intended as a diagnostic gate for proposal-generation
changes, not as an object-detection metric.
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
from common.metrics import get_metric_fn
from common.model import build_model, iterative_rpn_proposals

SIZE_BINS = (
    ("micro", 0.0, 8.0),
    ("tiny", 8.0, 16.0),
    ("small", 16.0, 32.0),
    ("large", 32.0, float("inf")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit RPN proposal recall by top-N and object size")
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument(
        "--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--transform-min-size",
        type=int,
        default=None,
        help="Override the checkpoint/model evaluation minimum image size",
    )
    parser.add_argument(
        "--transform-max-size",
        type=int,
        default=None,
        help="Override the checkpoint/model evaluation maximum image size",
    )
    parser.add_argument(
        "--top-n", type=int, nargs="+", default=(100, 300, 1000, 1500))
    parser.add_argument(
        "--iou-thresholds", type=float, nargs="+", default=(0.5, 0.75))
    parser.add_argument(
        "--rpn-refine-passes", type=int, default=1,
        help="Repeated applications of the fixed RPN box deltas")
    parser.add_argument(
        "--rpn-refine-min-size-ratio",
        type=float,
        default=0.0,
        help=(
            "Repeat deltas only above this normalized proposal sqrt-area; "
            "zero disables the gate"
        ),
    )
    parser.add_argument(
        "--rpn-iou-prediction-fusion-weight",
        type=float,
        default=None,
        help="Override the checkpoint's geometric presence-IoU fusion weight",
    )
    parser.add_argument(
        "--verify-pass1-parity", action="store_true",
        help="Require custom pass-1 proposals to match model.rpn exactly")
    parser.add_argument(
        "--max-tiles", type=int, default=None,
        help="Optional bounded smoke-test limit")
    parser.add_argument(
        "--out", default="runs/rpn_proposal_recall_valid.json")
    return parser.parse_args()


def build_checkpoint_model(
    checkpoint: dict,
    device: torch.device,
    iou_fusion_weight: float | None = None,
) -> torch.nn.Module:
    config = checkpoint.get("config", {})
    refine_blend = float(config.get("cbl_refine_blend", 1.0))
    refine_last_step_blend = config.get(
        "cbl_refine_last_step_blend")
    if refine_last_step_blend is None:
        refine_last_step_blend = refine_blend
    refine_last_center_blend = config.get(
        "cbl_refine_last_center_blend")
    if refine_last_center_blend is None:
        refine_last_center_blend = refine_last_step_blend
    refine_last_size_blend = config.get(
        "cbl_refine_last_size_blend")
    if refine_last_size_blend is None:
        refine_last_size_blend = refine_last_step_blend
    metric_name = config.get("metric", "sa_alw_full")
    placement = config.get("placement", "la_loss")
    if metric_name == "iou":
        metric_fn = None
        placement = "everywhere"
    else:
        metric_fn = get_metric_fn(metric_name)

    model = build_model(
        metric_fn=metric_fn,
        placement=placement,
        reliability_thr=float(config.get("reliability_thr", 16.0)),
        box_loss_type=config.get("box_loss", "metric"),
        box_loss_warmup_epochs=int(
            config.get("box_loss_warmup_epochs", 3)),
        use_quality_score=bool(config.get("quality_score", False)),
        quality_loss_weight=float(
            config.get("quality_loss_weight", 0.0) or 0.0),
        use_quality_focal=bool(config.get("quality_focal", False)),
        quality_focal_beta=float(config.get("quality_focal_beta", 2.0)),
        use_rank_sort=bool(config.get("rank_sort", False)),
        rank_sort_delta=float(config.get("rank_sort_delta", 0.5)),
        use_double_head=bool(config.get("double_head", False)),
        double_head_reg_roi_scale=float(
            config.get("double_head_reg_roi_scale", 1.3)),
        double_head_num_convs=int(config.get("double_head_num_convs", 4)),
        cbl_refine_steps=int(config.get("cbl_refine_steps", 0)),
        cbl_refine_blend=refine_blend,
        cbl_refine_last_step_blend=float(
            refine_last_step_blend),
        cbl_refine_last_center_blend=float(
            refine_last_center_blend),
        cbl_refine_last_size_blend=float(
            refine_last_size_blend),
        cbl_refine_score_threshold=float(
            config.get("cbl_refine_score_threshold", 0.0)),
        cbl_refine_extra_min_size_ratio=float(
            config.get("cbl_refine_extra_min_size_ratio", 0.0)),
        cbl_refine_train_weight=float(
            config.get("cbl_refine_train_weight", 0.0)),
        rpn_quality_objectness=bool(
            config.get("rpn_quality_objectness", False)),
        rpn_quality_beta=float(
            config.get("rpn_quality_beta", 2.0)),
        rpn_quality_preserve_below_size_ratio=float(
            config.get(
                "rpn_quality_preserve_below_size_ratio", 0.0)),
        rpn_cascade=bool(config.get("rpn_cascade", False)),
        rpn_cascade_stage1_weight=float(
            config.get("rpn_cascade_stage1_weight", 1.0)),
        rpn_iou_prediction=bool(
            config.get("rpn_iou_prediction", False)),
        rpn_iou_prediction_loss_weight=float(
            config.get("rpn_iou_prediction_loss_weight", 0.5)),
        rpn_iou_prediction_fusion_weight=(
            float(config.get(
                "rpn_iou_prediction_fusion_weight", 1.0))
            if iou_fusion_weight is None
            else iou_fusion_weight
        ),
        rpn_iou_prediction_detached_tower=bool(
            config.get("rpn_iou_prediction_detached_tower", False)),
        cbl_alpha=float(config.get("cbl_alpha", 5.0)),
        cbl_num_bins=int(config.get("cbl_num_bins", 6)),
        cbl_grid_beta=float(config.get("cbl_grid_beta", 1.0)),
        cbl_um_weight=float(config.get("cbl_um_weight", 1.0)),
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def size_bin_masks(boxes: torch.Tensor) -> dict[str, torch.Tensor]:
    widths = (boxes[:, 2] - boxes[:, 0]).clamp(min=0)
    heights = (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    sizes = torch.sqrt(widths * heights)
    return {
        name: (sizes >= lower) & (sizes < upper)
        for name, lower, upper in SIZE_BINS
    }


def empty_counter(
    top_ns: list[int], thresholds: list[float]
) -> dict[str, dict[str, int]]:
    return {
        f"top{top_n}": {
            f"iou{threshold:g}": 0 for threshold in thresholds
        }
        for top_n in top_ns
    }


def main() -> None:
    args = parse_args()
    seed_all(SEED)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.transform_min_size is not None and args.transform_min_size < 1:
        raise ValueError("--transform-min-size must be positive")
    if args.transform_max_size is not None and args.transform_max_size < 1:
        raise ValueError("--transform-max-size must be positive")
    top_ns = sorted(set(args.top_n))
    thresholds = sorted(set(args.iou_thresholds))
    if not top_ns or top_ns[0] < 1:
        raise ValueError("--top-n values must be positive")
    if not thresholds or any(not 0 <= value <= 1 for value in thresholds):
        raise ValueError("--iou-thresholds values must be in [0, 1]")
    if args.max_tiles is not None and args.max_tiles < 1:
        raise ValueError("--max-tiles must be positive")
    if args.rpn_refine_passes < 1:
        raise ValueError("--rpn-refine-passes must be positive")
    if args.rpn_refine_min_size_ratio < 0:
        raise ValueError("--rpn-refine-min-size-ratio must be non-negative")
    if (
        args.rpn_iou_prediction_fusion_weight is not None
        and not 0 <= args.rpn_iou_prediction_fusion_weight <= 1
    ):
        raise ValueError(
            "--rpn-iou-prediction-fusion-weight must be in [0, 1]")
    if args.verify_pass1_parity and args.rpn_refine_passes != 1:
        raise ValueError("--verify-pass1-parity requires exactly one pass")

    checkpoint_path = Path(args.ckpt)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu")

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

    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False)
    model = build_checkpoint_model(
        checkpoint,
        device,
        iou_fusion_weight=args.rpn_iou_prediction_fusion_weight,
    )
    if args.transform_min_size is not None:
        model.transform.min_size = (args.transform_min_size,)
    if args.transform_max_size is not None:
        model.transform.max_size = args.transform_max_size
    effective_transform_min_size = tuple(
        int(value) for value in model.transform.min_size)
    effective_transform_max_size = int(model.transform.max_size)
    config = checkpoint.get("config", {})
    cascade_enabled = bool(getattr(model.rpn, "cascade_refinement", False))
    iou_prediction_enabled = bool(
        getattr(model.rpn, "iou_prediction", False))
    native_rpn = (
        args.rpn_refine_passes == 1
        and args.rpn_refine_min_size_ratio == 0
    )
    if cascade_enabled and not native_rpn:
        raise ValueError(
            "Repeat-delta proposal audits are incompatible with RPN cascade")
    if cascade_enabled and args.verify_pass1_parity:
        raise ValueError(
            "Pass-1 helper parity is not defined for the learned RPN cascade")
    if iou_prediction_enabled and not native_rpn:
        raise ValueError(
            "Repeat-delta audits bypass RPN IoU-quality proposal ranking")
    if iou_prediction_enabled and args.verify_pass1_parity:
        raise ValueError(
            "Pass-1 helper parity is not defined for RPN IoU prediction")

    overall_hits = empty_counter(top_ns, thresholds)
    bin_hits = {
        name: empty_counter(top_ns, thresholds)
        for name, _, _ in SIZE_BINS
    }
    counts = {"overall": 0, **{name: 0 for name, _, _ in SIZE_BINS}}
    max_iou_samples = {top_n: [] for top_n in top_ns}
    num_tiles = 0
    num_tiles_with_gt = 0
    proposal_total = 0
    parity_max_abs_diff = 0.0

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
                        if torch.is_tensor(value) else value
                    )
                    for key, value in target.items()
                }
                for target in targets[:remaining]
            ]

            image_list, transformed_targets = model.transform(
                images, original_targets)
            features = model.backbone(image_list.tensors)
            if isinstance(features, torch.Tensor):
                features = OrderedDict([("0", features)])
            if native_rpn:
                proposals, _ = model.rpn(image_list, features)
            else:
                proposals = iterative_rpn_proposals(
                    model.rpn,
                    image_list,
                    features,
                    total_passes=args.rpn_refine_passes,
                    min_refine_size_ratio=args.rpn_refine_min_size_ratio,
                )
            if args.verify_pass1_parity:
                helper_proposals = iterative_rpn_proposals(
                    model.rpn,
                    image_list,
                    features,
                    total_passes=1,
                )
                for actual, reference in zip(
                    helper_proposals, proposals
                ):
                    if actual.shape != reference.shape:
                        raise AssertionError(
                            "Pass-1 proposal shape differs from model.rpn")
                    if actual.numel():
                        parity_max_abs_diff = max(
                            parity_max_abs_diff,
                            float((actual - reference).abs().max().item()),
                        )
                    if not torch.equal(actual, reference):
                        raise AssertionError(
                            "Pass-1 proposals differ from model.rpn")

            for proposal, original, transformed in zip(
                proposals, original_targets, transformed_targets
            ):
                num_tiles += 1
                proposal_total += len(proposal)
                gt_boxes = transformed["boxes"]
                original_boxes = original["boxes"]
                if gt_boxes.numel() == 0:
                    continue
                num_tiles_with_gt += 1
                counts["overall"] += len(gt_boxes)
                masks = size_bin_masks(original_boxes)
                for name, mask in masks.items():
                    counts[name] += int(mask.sum().item())

                for top_n in top_ns:
                    selected = proposal[:top_n]
                    if selected.numel() == 0:
                        max_iou = torch.zeros(
                            len(gt_boxes), device=gt_boxes.device)
                    else:
                        max_iou = box_iou(
                            gt_boxes, selected).max(dim=1).values
                    max_iou_samples[top_n].append(max_iou.cpu())
                    for threshold in thresholds:
                        threshold_key = f"iou{threshold:g}"
                        hits = max_iou >= threshold
                        overall_hits[f"top{top_n}"][threshold_key] += int(
                            hits.sum().item())
                        for name, mask in masks.items():
                            bin_hits[name][f"top{top_n}"][
                                threshold_key
                            ] += int((hits & mask).sum().item())

            if num_tiles % 200 < args.batch_size:
                print(
                    f"Audited {num_tiles}/{min(len(dataset), args.max_tiles or len(dataset))} "
                    f"tiles", flush=True)

    def recalls(
        hits: dict[str, dict[str, int]], denominator: int
    ) -> dict[str, dict[str, float]]:
        return {
            top_key: {
                threshold_key: (
                    value / denominator if denominator else 0.0)
                for threshold_key, value in threshold_hits.items()
            }
            for top_key, threshold_hits in hits.items()
        }

    quantiles = {}
    for top_n, samples in max_iou_samples.items():
        values = torch.cat(samples) if samples else torch.empty(0)
        quantiles[f"top{top_n}"] = {
            key: float(value)
            for key, value in zip(
                ("min", "p10", "p25", "median", "p75", "p90", "max"),
                torch.quantile(
                    values,
                    torch.tensor(
                        (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
                        dtype=values.dtype,
                    ),
                ).tolist() if values.numel() else (0.0,) * 7,
            )
        }

    result = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_model_source": checkpoint.get(
            "model_source", "legacy_unspecified"),
        "split": args.split,
        "metric": config.get("metric", "sa_alw_full"),
        "placement": config.get("placement", "la_loss"),
        "transform_min_size": list(effective_transform_min_size),
        "transform_max_size": effective_transform_max_size,
        "num_tiles": num_tiles,
        "num_tiles_with_gt": num_tiles_with_gt,
        "num_gt": counts["overall"],
        "mean_proposals_per_tile": (
            proposal_total / num_tiles if num_tiles else 0.0),
        "rpn_refine_passes": args.rpn_refine_passes,
        "rpn_refine_min_size_ratio": (
            args.rpn_refine_min_size_ratio),
        "rpn_cascade": cascade_enabled,
        "rpn_cascade_stage1_weight": float(
            config.get("rpn_cascade_stage1_weight", 1.0)),
        "rpn_iou_prediction": iou_prediction_enabled,
        "rpn_iou_prediction_loss_weight": float(
            config.get("rpn_iou_prediction_loss_weight", 0.5)),
        "rpn_iou_prediction_fusion_weight": float(
            getattr(
                model.rpn,
                "iou_prediction_fusion_weight",
                1.0,
            )
        ),
        "rpn_iou_prediction_detached_tower": bool(
            config.get("rpn_iou_prediction_detached_tower", False)),
        "pass1_parity_verified": args.verify_pass1_parity,
        "pass1_parity_max_abs_diff": (
            parity_max_abs_diff if args.verify_pass1_parity else None),
        "top_n": top_ns,
        "iou_thresholds": thresholds,
        "size_bins_sqrt_area_px": {
            name: [lower, None if upper == float("inf") else upper]
            for name, lower, upper in SIZE_BINS
        },
        "gt_counts": counts,
        "recall": {
            "overall": recalls(overall_hits, counts["overall"]),
            "by_size": {
                name: recalls(bin_hits[name], counts[name])
                for name, _, _ in SIZE_BINS
            },
        },
        "max_iou_quantiles": quantiles,
    }

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(result["recall"], indent=2))
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
