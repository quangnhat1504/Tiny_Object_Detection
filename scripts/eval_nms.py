"""
Evaluate la_loss_nms placement using existing la_loss weights.

Metric-NMS replaces IoU-NMS during inference — no retraining needed.
Simply rebuild model with placement="la_loss_nms", load la_loss weights, eval.

Usage:
    python scripts/eval_nms.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import SEED, seed_all, DEVICE
from common.dataset import YOLOTinyDataset, collate_fn, compute_reliability_threshold
from common.metrics import get_metric_fn, NEEDS_RELIABILITY
from common.model import build_model
from common.eval_utils import evaluate

RUNS_DIR = ROOT / "runs"

# Metrics to evaluate with metric-NMS
TARGETS = [
    {"metric": "alw_full", "seed": 42},
    {"metric": "sa_alw_full", "seed": 42},
    {"metric": "sa_alw_pos_only", "seed": 42},
    {"metric": "sa_alw_beta_only", "seed": 42},
]


def build_test_loader():
    td = ROOT / "data" / "test"
    ds = YOLOTinyDataset(img_dir=td / "images", lbl_dir=td / "labels", is_train=False)
    return DataLoader(ds, batch_size=2, shuffle=False, num_workers=0,
                      collate_fn=collate_fn, pin_memory=(DEVICE.type == "cuda"))


def eval_nms(metric_name: str, seed: int, test_loader):
    run_name = f"{metric_name}__la_loss__seed{seed}"
    ckpt_path = RUNS_DIR / run_name / "best.pt"
    if not ckpt_path.exists():
        print(f"  SKIP: no checkpoint at {ckpt_path}")
        return None

    print(f"\n{'='*60}")
    print(f"  {metric_name} @ la_loss_nms  (weights from la_loss)")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stored = ck.get("config", {})
    reliability_thr = stored.get("reliability_thr", 16.0)

    metric_fn = get_metric_fn(metric_name)
    model = build_model(metric_fn=metric_fn, placement="la_loss_nms",
                        reliability_thr=reliability_thr).to(DEVICE)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    # Lower score thresh for test — la_loss_nms needs to see more candidates
    model.roi_heads.score_thresh = 0.001

    metrics = evaluate(model, test_loader, DEVICE, measure_fps_flag=False)

    result = {
        "name": f"{metric_name}__la_loss_nms__seed{seed}",
        "metric": metric_name,
        "placement": "la_loss_nms",
        "seed": seed,
        "ckpt_source": f"{metric_name}__la_loss__seed{seed}/best.pt",
        "ckpt_epoch": ck.get("epoch", "unknown"),
        "val_best_mAP50": round(ck.get("best_mAP50", ck.get("best_metric_value", 0.0)) or 0.0, 4),
        "test": metrics,
    }

    print(f"  mAP(scale): {metrics.get('mAP_primary', 0):.4f}")
    print(f"  AP_micro:   {metrics.get('AP_micro', 0):.4f}")
    print(f"  AP_tiny:    {metrics.get('AP_tiny', 0):.4f}")
    print(f"  mAP@50:     {metrics.get('mAP_50', 0):.4f}")
    return result


def main():
    seed_all(SEED)
    print(f"Device: {DEVICE}")

    test_loader = build_test_loader()
    print(f"Test set: {len(test_loader.dataset)} tiles\n")

    # Load existing la_loss results for comparison
    agg_path = RUNS_DIR / "test_results.json"
    la_loss_results = {}
    if agg_path.exists():
        with open(agg_path) as f:
            agg = json.load(f)
        for r in agg["results"]:
            if r.get("placement") == "la_loss":
                la_loss_results[r["metric"]] = r["test"]

    nms_results = []
    for t in TARGETS:
        res = eval_nms(t["metric"], t["seed"], test_loader)
        if res:
            nms_results.append(res)

            # Save per-run
            out_dir = RUNS_DIR / f"{t['metric']}__la_loss_nms__seed{t['seed']}"
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "test_metrics.json", "w") as f:
                json.dump(res, f, indent=2)

    # Comparison table
    print("\n\n" + "=" * 110)
    print("la_loss  ->  la_loss_nms  COMPARISON (test set)")
    print("=" * 110)
    print(f"{'Metric':<25} {'Placement':>12} {'mAP(scale)':>10} {'AP_micro':>9} {'AP_tiny':>9} {'AP_small':>9} {'AP_large':>9} {'mAP@50':>8} {'d_mAP(s)':>9}")
    print("-" * 110)

    for nr in nms_results:
        m = nr["metric"]
        t_nms = nr["test"]
        t_la = la_loss_results.get(m, {})

        delta = t_nms.get("mAP_primary", 0) - t_la.get("mAP_primary", 0)
        d_sign = "+" if delta > 0 else ""

        for label, tdata in [("la_loss", t_la), ("la_loss_nms", t_nms)]:
            if not tdata:
                continue
            row = (f"{m:<25} {label:>12} "
                   f"{tdata.get('mAP_primary', 0):>10.4f} "
                   f"{tdata.get('AP_micro', 0):>9.4f} "
                   f"{tdata.get('AP_tiny', 0):>9.4f} "
                   f"{tdata.get('AP_small', 0):>9.4f} "
                   f"{tdata.get('AP_large', 0):>9.4f} "
                   f"{tdata.get('mAP_50', 0):>8.4f}")
            if label == "la_loss_nms":
                row += f"  {d_sign}{delta:+.4f}"
            print(row)
        print()

    print("-" * 110)


if __name__ == "__main__":
    main()
