"""Direct single-checkpoint evaluation on test or validation split.

Usage:
    python scripts/test_coco_eval_single.py --split test --device cuda \
        --ckpt runs/sa_alw_full__la_loss__seed42/best.pt \
        --out runs/test_coco_sa_alw_full.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import SEED, seed_all, DEVICE
from common.dataset import YOLOTinyDataset, collate_fn
from common.metrics import get_metric_fn, NEEDS_RELIABILITY
from common.model import build_model
from common.eval_utils import evaluate


def main():
    parser = argparse.ArgumentParser(description="Direct COCO evaluation on a single checkpoint")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Path to checkpoint .pt file")
    parser.add_argument("--metric", type=str, default="sa_alw_full",
                        help="Metric name (default: sa_alw_full)")
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="test",
                        help="Dataset split (default: test)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device (default: cuda)")
    parser.add_argument("--cbl-refine-steps", type=int, default=None,
                        help="Inference-only repeated CBL box-regression passes")
    parser.add_argument("--cbl-refine-blend", type=float, default=None,
                        help="Fraction of each iterative CBL box update")
    parser.add_argument(
        "--cbl-refine-last-step-blend",
        type=float,
        default=None,
        help="Fraction of only the final iterative CBL box update",
    )
    parser.add_argument(
        "--cbl-refine-last-center-blend",
        type=float,
        default=None,
        help="Fraction of only the final center update",
    )
    parser.add_argument(
        "--cbl-refine-last-size-blend",
        type=float,
        default=None,
        help="Fraction of only the final width/height update",
    )
    parser.add_argument("--cbl-refine-score-threshold", type=float, default=None,
                        help="Preserve detections below this score")
    parser.add_argument(
        "--cbl-refine-extra-min-size-ratio",
        type=float,
        default=None,
        help=(
            "After pass one, refine only boxes at or above this normalized "
            "sqrt-area size; zero disables the gate"
        ),
    )
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: runs/test_coco_<metric>.json)")
    args = parser.parse_args()

    seed_all(SEED)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Split: {args.split}")

    # Build data loader
    data_dir = ROOT / "data" / args.split
    ds = YOLOTinyDataset(
        img_dir=data_dir / "images",
        lbl_dir=data_dir / "labels",
        is_train=False,
    )
    loader = DataLoader(
        ds, batch_size=2, shuffle=False, num_workers=0,
        collate_fn=collate_fn, pin_memory=(device.type == "cuda"),
    )
    print(f"{args.split} set: {len(ds)} tiles")

    # Load checkpoint
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stored_config = ck.get("config", {})
    effective_metric = stored_config.get("metric", args.metric)
    effective_placement = stored_config.get("placement", "la_loss")
    reliability_thr = stored_config.get("reliability_thr", 16.0)
    checkpoint_model_source = ck.get("model_source", "legacy_unspecified")
    stored_metrics_source = (
        "ema" if stored_config.get("use_ema", False) else "raw")
    refine_steps = (
        int(stored_config.get("cbl_refine_steps", 0))
        if args.cbl_refine_steps is None else args.cbl_refine_steps
    )
    refine_blend = (
        float(stored_config.get("cbl_refine_blend", 1.0))
        if args.cbl_refine_blend is None else args.cbl_refine_blend
    )
    refine_last_step_blend = (
        stored_config.get("cbl_refine_last_step_blend")
        if args.cbl_refine_last_step_blend is None
        else args.cbl_refine_last_step_blend
    )
    if refine_last_step_blend is None:
        refine_last_step_blend = refine_blend
    refine_last_step_blend = float(refine_last_step_blend)
    refine_last_center_blend = (
        stored_config.get("cbl_refine_last_center_blend")
        if args.cbl_refine_last_center_blend is None
        else args.cbl_refine_last_center_blend
    )
    if refine_last_center_blend is None:
        refine_last_center_blend = refine_last_step_blend
    refine_last_center_blend = float(refine_last_center_blend)
    refine_last_size_blend = (
        stored_config.get("cbl_refine_last_size_blend")
        if args.cbl_refine_last_size_blend is None
        else args.cbl_refine_last_size_blend
    )
    if refine_last_size_blend is None:
        refine_last_size_blend = refine_last_step_blend
    refine_last_size_blend = float(refine_last_size_blend)
    refine_score_threshold = (
        float(stored_config.get("cbl_refine_score_threshold", 0.0))
        if args.cbl_refine_score_threshold is None
        else args.cbl_refine_score_threshold
    )
    refine_extra_min_size_ratio = (
        float(stored_config.get("cbl_refine_extra_min_size_ratio", 0.0))
        if args.cbl_refine_extra_min_size_ratio is None
        else args.cbl_refine_extra_min_size_ratio
    )

    print(f"Stored config: metric={effective_metric}, placement={effective_placement}")
    print(f"Checkpoint epoch: {ck.get('epoch', 'unknown')}")
    print(
        f"Checkpoint model source: {checkpoint_model_source}; "
        f"stored metrics source: {stored_metrics_source}")
    if checkpoint_model_source != stored_metrics_source:
        print(
            "WARNING: stored metrics may not describe checkpoint['model']; "
            "use the independent evaluation below.")
    print(f"Val best mAP50: {round(ck.get('best_mAP50', ck.get('best_metric_value', 0.0)) or 0.0, 4)}")

    # Build metric function
    if effective_metric == "iou":
        metric_fn = None
        effective_placement = "everywhere"
    else:
        metric_fn = get_metric_fn(effective_metric)

    # Build model
    model = build_model(
        metric_fn=metric_fn,
        placement=effective_placement,
        reliability_thr=reliability_thr,
        box_loss_type=stored_config.get("box_loss", "metric"),
        box_loss_warmup_epochs=int(
            stored_config.get("box_loss_warmup_epochs", 3)),
        use_quality_score=bool(stored_config.get("quality_score", False)),
        quality_loss_weight=float(stored_config.get("quality_loss_weight", 0.0) or 0.0),
        use_quality_focal=bool(stored_config.get("quality_focal", False)),
        quality_focal_beta=float(stored_config.get("quality_focal_beta", 2.0)),
        use_rank_sort=bool(stored_config.get("rank_sort", False)),
        rank_sort_delta=float(stored_config.get("rank_sort_delta", 0.5)),
        use_double_head=bool(stored_config.get("double_head", False)),
        double_head_reg_roi_scale=float(
            stored_config.get("double_head_reg_roi_scale", 1.3)),
        double_head_num_convs=int(
            stored_config.get("double_head_num_convs", 4)),
        cbl_refine_steps=refine_steps,
        cbl_refine_blend=refine_blend,
        cbl_refine_last_step_blend=refine_last_step_blend,
        cbl_refine_last_center_blend=refine_last_center_blend,
        cbl_refine_last_size_blend=refine_last_size_blend,
        cbl_refine_score_threshold=refine_score_threshold,
        cbl_refine_extra_min_size_ratio=refine_extra_min_size_ratio,
        cbl_refine_train_weight=float(
            stored_config.get("cbl_refine_train_weight", 0.0)),
        cbl_alpha=float(stored_config.get("cbl_alpha", 5.0)),
        cbl_num_bins=int(stored_config.get("cbl_num_bins", 6)),
        cbl_grid_beta=float(stored_config.get("cbl_grid_beta", 1.0)),
        cbl_um_weight=float(stored_config.get("cbl_um_weight", 1.0)),
    ).to(device)

    # Load weights
    model.load_state_dict(ck["model"])
    model.eval()

    # Evaluate
    print(f"\nRunning evaluation...")
    metrics = evaluate(model, loader, device, measure_fps_flag=False)

    # Extract key COCO metrics
    coco_ap = metrics.get("coco_AP", 0)
    coco_ap50 = metrics.get("coco_AP50", 0)
    coco_ap75 = metrics.get("coco_AP75", 0)
    coco_ar100 = metrics.get("coco_AR100", 0)

    print(f"\n{'='*50}")
    print(f"RESULTS:")
    print(f"  COCO AP     : {coco_ap:.4f}")
    print(f"  COCO AP@50  : {coco_ap50:.4f}")
    print(f"  COCO AP@75  : {coco_ap75:.4f}")
    print(f"  COCO AR@100 : {coco_ar100:.4f}")
    print(f"{'='*50}")

    # Save
    out_path = Path(args.out) if args.out else ROOT / "runs" / f"test_coco_{effective_metric}.json"
    result = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "metric": effective_metric,
        "placement": effective_placement,
        "cbl_refine_steps": refine_steps,
        "cbl_refine_blend": refine_blend,
        "cbl_refine_last_step_blend": refine_last_step_blend,
        "cbl_refine_last_center_blend": refine_last_center_blend,
        "cbl_refine_last_size_blend": refine_last_size_blend,
        "cbl_refine_score_threshold": refine_score_threshold,
        "cbl_refine_extra_min_size_ratio": refine_extra_min_size_ratio,
        "ckpt_epoch": ck.get("epoch", "unknown"),
        "checkpoint_model_source": checkpoint_model_source,
        "stored_metrics_source": stored_metrics_source,
        "stored_metrics_match_checkpoint": (
            checkpoint_model_source == stored_metrics_source),
        "val_best_mAP50": round(ck.get("best_mAP50", ck.get("best_metric_value", 0.0)) or 0.0, 4),
        "coco_AP": round(coco_ap, 4),
        "coco_AP50": round(coco_ap50, 4),
        "coco_AP75": round(coco_ap75, 4),
        "coco_AP_small": round(metrics.get("coco_AP_small", 0), 4),
        "coco_AP_medium": round(metrics.get("coco_AP_medium", 0), 4),
        "coco_AP_large": round(metrics.get("coco_AP_large", 0), 4),
        "coco_AR1": round(metrics.get("coco_AR1", 0), 4),
        "coco_AR10": round(metrics.get("coco_AR10", 0), 4),
        "coco_AR100": round(coco_ar100, 4),
        "mAP_primary": round(metrics.get("mAP_primary", 0), 4),
        "mAP_50": round(metrics.get("mAP_50", 0), 4),
        "AP_micro": metrics.get("AP_micro", 0),
        "AP_tiny": metrics.get("AP_tiny", 0),
        "AP_small": metrics.get("AP_small", 0),
        "AP_large": metrics.get("AP_large", 0),
        "full_metrics": metrics,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
