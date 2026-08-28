"""Sweep score_thresh and nms_thresh on valid. One inference pass, post-filter configs.

Usage:
    python scripts/eval_direct_configs.py --split valid --device cuda \
        --ckpt runs/sa_alw_full__la_loss__seed42/best.pt \
        --out runs/direct_configs_valid_sa_alw_full.json
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision.ops import nms

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import SEED, seed_all, DEVICE
from common.dataset import YOLOTinyDataset, collate_fn
from common.metrics import get_metric_fn
from common.model import build_model
from common.eval_utils import evaluate_coco, compute_scale_ap, compute_precision_recall


@torch.no_grad()
def collect_all_boxes(model, loader, device, amp: bool = True):
    """Collect box proposals with minimal score/NMS filtering, then post-filter.

    Returns raw predictions before NMS filtering threshold.
    We set score_thresh=0.001 and nms_thresh=1.0 to get nearly all candidates,
    then apply the requested score/NMS thresholds post-hoc.
    """
    model.eval()
    # Lower score threshold to capture all candidates
    prev_score = model.roi_heads.score_thresh
    model.roi_heads.score_thresh = 0.001
    prev_nms = model.roi_heads.nms_thresh
    model.roi_heads.nms_thresh = 1.0
    # Keep high max detections
    prev_det = model.roi_heads.detections_per_img
    model.roi_heads.detections_per_img = 1000

    preds_all, gts_all = [], []
    for imgs, targets in loader:
        try:
            with torch.amp.autocast("cuda", enabled=amp):
                p = model([i.to(device) for i in imgs])
            for pp, tt in zip(p, targets):
                preds_all.append({k: v.cpu() for k, v in pp.items()})
                gts_all.append({k: v.cpu() if isinstance(v, torch.Tensor) else v
                                for k, v in tt.items()})
        except Exception:
            continue

    model.roi_heads.score_thresh = prev_score
    model.roi_heads.nms_thresh = prev_nms
    model.roi_heads.detections_per_img = prev_det
    return preds_all, gts_all


def filter_preds(preds: List[Dict], score_thr: float, nms_thr: float) -> List[Dict]:
    """Post-filter predictions by score threshold and apply NMS."""
    filtered = []
    for p in preds:
        boxes = p["boxes"]
        scores = p["scores"]
        labels = p["labels"]

        # Score filter
        keep = scores >= score_thr
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

        if len(boxes) == 0:
            filtered.append({"boxes": torch.empty(0, 4), "scores": torch.empty(0),
                             "labels": torch.empty(0, dtype=torch.int64)})
            continue

        # NMS
        keep_idx = nms(boxes, scores, nms_thr)
        filtered.append({
            "boxes": boxes[keep_idx],
            "scores": scores[keep_idx],
            "labels": labels[keep_idx],
        })
    return filtered


def compute_metrics_dict(preds, gts) -> Dict:
    coco = evaluate_coco(preds, gts, class_metrics=False)
    sap = compute_scale_ap(preds, gts)
    tgt = sum(sap.get(f"n_gt_{s}", 0) for s in ("micro", "tiny", "small", "large"))
    mAP_primary = (sum(sap.get(f"AP_{s}", 0.0) * sap.get(f"n_gt_{s}", 0)
                       for s in ("micro", "tiny", "small", "large")) / tgt
                   if tgt > 0 else 0.0)
    pr = compute_precision_recall(preds, gts, iou_thresh=0.5, score_thresh=0.05)
    return {
        "coco_AP": round(coco.get("coco_AP", 0), 4),
        "coco_AP50": round(coco.get("coco_AP50", 0), 4),
        "coco_AP75": round(coco.get("coco_AP75", 0), 4),
        "coco_AR100": round(coco.get("coco_AR100", 0), 4),
        "coco_AR10": round(coco.get("coco_AR10", 0), 4),
        "coco_AR1": round(coco.get("coco_AR1", 0), 4),
        "mAP_primary": round(mAP_primary, 4),
        "AP_micro": sap.get("AP_micro", 0),
        "AP_tiny": sap.get("AP_tiny", 0),
        "AP_small": sap.get("AP_small", 0),
        "AP_large": sap.get("AP_large", 0),
        "Precision": pr.get("Precision", 0),
        "Recall": pr.get("Recall", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Sweep direct eval configs on valid/test")
    parser.add_argument("--split", type=str, choices=["valid", "test"], default="valid")
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--score-thrs", type=float, nargs="+",
                        default=[0.05, 0.10, 0.20, 0.30, 0.40])
    parser.add_argument("--nms-thrs", type=float, nargs="+",
                        default=[0.30, 0.40, 0.50, 0.60, 0.70])
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--metric", type=str, default="sa_alw_full")
    args = parser.parse_args()

    seed_all(SEED)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stored_config = ck.get("config", {})
    effective_metric = stored_config.get("metric", args.metric)
    effective_placement = stored_config.get("placement", "la_loss")
    reliability_thr = stored_config.get("reliability_thr", 16.0)

    print(f"Stored config: metric={effective_metric}, placement={effective_placement}")
    print(f"Checkpoint epoch: {ck.get('epoch', 'unknown')}")

    metric_fn = None if effective_metric == "iou" else get_metric_fn(effective_metric)

    # Data loader
    data_dir = ROOT / "data" / args.split
    ds = YOLOTinyDataset(img_dir=data_dir / "images", lbl_dir=data_dir / "labels", is_train=False)
    loader = DataLoader(ds, batch_size=2, shuffle=False, num_workers=0,
                        collate_fn=collate_fn, pin_memory=(device.type == "cuda"))
    print(f"{args.split} set: {len(ds)} tiles")

    # Build model once
    model = build_model(
        metric_fn=metric_fn, placement=effective_placement,
        reliability_thr=reliability_thr,
        box_loss_type=stored_config.get("box_loss", "metric"),
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
        cbl_alpha=float(stored_config.get("cbl_alpha", 5.0)),
        cbl_num_bins=int(stored_config.get("cbl_num_bins", 6)),
        cbl_grid_beta=float(stored_config.get("cbl_grid_beta", 1.0)),
        cbl_um_weight=float(stored_config.get("cbl_um_weight", 1.0)),
    ).to(device)
    model.load_state_dict(ck["model"])

    default_score = model.roi_heads.score_thresh
    default_nms = model.roi_heads.nms_thresh
    print(f"Model defaults: score_thresh={default_score}, nms_thresh={default_nms}")

    # ONE inference pass: collect all proposals at score_thresh=0.001
    print("\nCollecting all box proposals (score_thresh=0.001)...")
    raw_preds, gts = collect_all_boxes(model, loader, device)

    # Count total boxes
    total_boxes = sum(len(p["boxes"]) for p in raw_preds)
    print(f"Collected {total_boxes} boxes across {len(raw_preds)} tiles")

    del model
    torch.cuda.empty_cache()

    # Evaluate default config with post-filtering
    print("\nDefault config...")
    def_preds = filter_preds(raw_preds, default_score, default_nms)
    def_met = compute_metrics_dict(def_preds, gts)
    def_met["score_thr"] = default_score
    def_met["nms_thr"] = default_nms
    print(f"  AP75={def_met['coco_AP75']} AP={def_met['coco_AP']} AR100={def_met['coco_AR100']}")

    # Sweep
    results: List[Dict] = []
    total_configs = len(args.nms_thrs) * len(args.score_thrs)
    config_idx = 0

    for nms_thr in args.nms_thrs:
        for score_thr in args.score_thrs:
            config_idx += 1
            print(f"[{config_idx}/{total_configs}] score={score_thr} nms={nms_thr}")

            fpreds = filter_preds(raw_preds, score_thr, nms_thr)
            met = compute_metrics_dict(fpreds, gts)
            met["score_thr"] = score_thr
            met["nms_thr"] = nms_thr
            results.append(met)
            print(f"  AP75={met['coco_AP75']} AP={met['coco_AP']} AR100={met['coco_AR100']}")

    # Sort
    results.sort(key=lambda r: (r["coco_AP75"], r["coco_AP"], r["coco_AR100"]), reverse=True)

    best_ap75 = results[0]["coco_AP75"] if results else 0
    beating = [r for r in results if r["coco_AP75"] > def_met["coco_AP75"]]

    print(f"\nBest sweep AP75: {best_ap75}  (default: {def_met['coco_AP75']})")
    print(f"Configs beating default: {len(beating)}")

    print(f"\n{'='*70}")
    print("TOP 10 by coco_AP75:")
    print(f"{'Rank':<5} {'score':>7} {'nms':>6} {'AP75':>8} {'AP':>8} {'AR100':>8} {'mAP(s)':>8}")
    print("-" * 70)
    for i, r in enumerate(results[:10]):
        print(f"{i+1:<5} {r['score_thr']:>7.2f} {r['nms_thr']:>6.2f} "
              f"{r['coco_AP75']:>8.4f} {r['coco_AP']:>8.4f} "
              f"{r['coco_AR100']:>8.4f} {r['mAP_primary']:>8.4f}")
    print(f"\n{'DEFAULT':<5} {def_met['score_thr']:>7.2f} {def_met['nms_thr']:>6.2f} "
          f"{def_met['coco_AP75']:>8.4f} {def_met['coco_AP']:>8.4f} "
          f"{def_met['coco_AR100']:>8.4f} {def_met['mAP_primary']:>8.4f}")

    out_path = Path(args.out) if args.out else ROOT / "runs" / f"direct_configs_{args.split}_{effective_metric}.json"
    output = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "metric": effective_metric,
        "placement": effective_placement,
        "ckpt_epoch": ck.get("epoch", "unknown"),
        "note": "score_thresh and nms_thr are post-filtered after collection with forward score_thresh=0.001 and nms_thresh=1.0.",
        "default": def_met,
        "results": results,
        "n_beating_default_AP75": len(beating),
        "best_frozen_config": {
            "score_thr": results[0]["score_thr"],
            "nms_thr": results[0]["nms_thr"],
            "coco_AP75": results[0]["coco_AP75"],
            "coco_AP": results[0]["coco_AP"],
            "coco_AR100": results[0]["coco_AR100"],
        } if results else None,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
