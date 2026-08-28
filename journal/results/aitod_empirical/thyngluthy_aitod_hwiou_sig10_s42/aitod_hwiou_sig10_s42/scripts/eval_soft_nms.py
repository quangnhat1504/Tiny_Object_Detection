"""
Evaluate ALW Soft-NMS (Phase 5) — ALW-based score decay instead of hard NMS.

No retraining needed — rebuild model with placement="la_loss_soft_nms",
load existing la_loss weights, evaluate on test set.

Usage:
    python scripts/eval_soft_nms.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import SEED, seed_all, DEVICE
from common.dataset import YOLOTinyDataset, collate_fn
from common.metrics import get_metric_fn
from common.model import build_model
from common.eval_utils import evaluate

RUNS_DIR = ROOT / "runs"

TARGETS = [
    {"metric": "alw_full", "seed": 42},
    {"metric": "sa_alw_full", "seed": 42},
]


def build_test_loader():
    td = ROOT / "data" / "test"
    ds = YOLOTinyDataset(img_dir=td / "images", lbl_dir=td / "labels", is_train=False)
    return DataLoader(ds, batch_size=2, shuffle=False, num_workers=0,
                      collate_fn=collate_fn, pin_memory=(DEVICE.type == "cuda"))


def eval_soft_nms(metric_name: str, seed: int, test_loader):
    run_name = f"{metric_name}__la_loss__seed{seed}"
    ckpt_path = RUNS_DIR / run_name / "best.pt"
    if not ckpt_path.exists():
        print(f"  SKIP: no checkpoint at {ckpt_path}")
        return None

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    stored = ck.get("config", {})
    reliability_thr = stored.get("reliability_thr", 16.0)

    metric_fn = get_metric_fn(metric_name)
    model = build_model(metric_fn=metric_fn, placement="la_loss_soft_nms",
                        reliability_thr=reliability_thr).to(DEVICE)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()
    model.roi_heads.score_thresh = 0.001

    metrics = evaluate(model, test_loader, DEVICE, measure_fps_flag=False)

    result = {
        "name": f"{metric_name}__la_loss_soft_nms__seed{seed}",
        "metric": metric_name,
        "placement": "la_loss_soft_nms",
        "seed": seed,
        "ckpt_source": f"{metric_name}__la_loss__seed{seed}/best.pt",
        "ckpt_epoch": ck.get("epoch", "unknown"),
        "test": metrics,
    }

    print(f"  mAP(scale): {metrics.get('mAP_primary', 0):.4f}")
    print(f"  AP_micro:   {metrics.get('AP_micro', 0):.4f}")
    return result


def main():
    seed_all(SEED)

    test_loader = build_test_loader()
    print(f"Test tiles: {len(test_loader.dataset)}")

    # Load existing la_loss results
    agg_path = RUNS_DIR / "test_results.json"
    la_loss_results = {}
    if agg_path.exists():
        agg = json.load(open(agg_path))
        for r in agg["results"]:
            if r.get("placement") == "la_loss":
                la_loss_results[r["metric"]] = r["test"]

    results = []
    for t in TARGETS:
        print(f"\n{'='*60}")
        print(f"  {t['metric']} @ la_loss_soft_nms")
        res = eval_soft_nms(t["metric"], t["seed"], test_loader)
        if res:
            results.append(res)
            out_dir = RUNS_DIR / f"{t['metric']}__la_loss_soft_nms__seed{t['seed']}"
            out_dir.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(out_dir / "test_metrics.json", "w"), indent=2)

    # Comparison table
    print("\n" + "=" * 100)
    print("standard NMS  ->  ALW Soft-NMS  COMPARISON (test set)")
    print("=" * 100)
    header = f"{'Metric':<20} {'Placement':>18} {'mAP(s)':>8} {'micro':>8} {'tiny':>8} {'small':>8} {'large':>8} {'mAP@50':>8} {'delta_mAP':>9}"
    print(header)
    print("-" * 100)

    for r in results:
        m = r["metric"]
        t_soft = r["test"]
        t_std = la_loss_results.get(m, {})

        for label, tdata in [("la_loss", t_std), ("la_loss_soft_nms", t_soft)]:
            if not tdata:
                continue
            delta = ""
            if label == "la_loss_soft_nms" and t_std:
                d = tdata.get("mAP_primary", 0) - t_std.get("mAP_primary", 0)
                sign = "+" if d > 0 else ""
                delta = f"  {sign}{d:+.4f}"
            print(f"{m:<20} {label:>18} "
                  f"{tdata.get('mAP_primary',0):>8.4f} "
                  f"{tdata.get('AP_micro',0):>8.4f} "
                  f"{tdata.get('AP_tiny',0):>8.4f} "
                  f"{tdata.get('AP_small',0):>8.4f} "
                  f"{tdata.get('AP_large',0):>8.4f} "
                  f"{tdata.get('mAP_50',0):>8.4f}"
                  f"{delta}")
        print()

    with open(RUNS_DIR / "soft_nms_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: runs/soft_nms_results.json")


if __name__ == "__main__":
    main()
