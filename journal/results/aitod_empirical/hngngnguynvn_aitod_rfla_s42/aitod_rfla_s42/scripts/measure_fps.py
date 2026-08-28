"""
Quick FPS measurement for all checkpoints using dummy input.

Usage:
    python scripts/measure_fps.py
    python scripts/measure_fps.py --metric alw_full
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import DEVICE, SEED, seed_all, TEST_DIR, NUM_CLASSES
from common.metrics import get_metric_fn, NEEDS_RELIABILITY
from common.model import build_model
from common.eval_utils import measure_fps

RUNS_DIR = ROOT / "runs"


def discover_checkpoints():
    entries = []
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        if run_dir.name.startswith("yolo_"):
            continue
        best_pt = run_dir / "best.pt"
        last_pt = run_dir / "last.pt"
        if not best_pt.exists() and not last_pt.exists():
            continue
        ckpt = best_pt if best_pt.exists() else last_pt

        if run_dir.name.startswith("frcnn_standard"):
            parts = run_dir.name.split("__")
            entry = {
                "name": run_dir.name,
                "checkpoint": str(ckpt),
                "metric": "iou",
                "placement": "everywhere",
                "seed": int(parts[2].replace("seed", "")),
            }
        else:
            parts = run_dir.name.split("__")
            entry = {
                "name": run_dir.name,
                "checkpoint": str(ckpt),
                "metric": parts[0],
                "placement": parts[1],
                "seed": int(parts[2].replace("seed", "")),
            }
        entries.append(entry)
    return entries


def measure_one(entry):
    print(f"\n  {entry['name']}")
    ck = torch.load(entry["checkpoint"], map_location="cpu", weights_only=False)
    stored = ck.get("config", {})

    metric_name = stored.get("metric", entry["metric"])
    placement = stored.get("placement", entry["placement"])

    if metric_name == "iou" or entry["metric"] == "iou":
        metric_fn = None
        placement = "everywhere"
    else:
        metric_fn = get_metric_fn(metric_name)

    reliability_thr = 16.0
    if metric_name in NEEDS_RELIABILITY:
        reliability_thr = stored.get("reliability_thr", 16.0)

    model = build_model(
        metric_fn=metric_fn,
        placement=placement,
        reliability_thr=reliability_thr,
    ).to(DEVICE)
    model.load_state_dict(ck["model"], strict=False)
    model.eval()

    fps = measure_fps(model, DEVICE, batch_size=1, n_warmup=30, n_iters=100)
    print(f"    FPS: {fps:.1f} ({1000/fps:.1f} ms/img)")
    return fps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default=None)
    args = parser.parse_args()

    seed_all(SEED)
    print(f"Device: {DEVICE}")

    entries = discover_checkpoints()
    if args.metric:
        entries = [e for e in entries if e.get("metric") == args.metric]

    print(f"Found {len(entries)} checkpoints\n")

    results = {}
    for e in entries:
        try:
            fps = measure_one(e)
        except Exception as ex:
            print(f"    ERROR: {ex}")
            fps = 0.0
        results[e["name"]] = round(fps, 1)

    # Update per-run test_metrics.json with FPS
    for e in entries:
        json_path = RUNS_DIR / e["name"] / "test_metrics.json"
        if json_path.exists():
            with open(json_path) as f:
                data = json.load(f)
            data["fps"] = results[e["name"]]
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)

    # Update aggregate
    agg_path = RUNS_DIR / "test_results.json"
    if agg_path.exists():
        with open(agg_path) as f:
            agg = json.load(f)
        for r in agg.get("results", []):
            r["fps"] = results.get(r["name"], 0.0)
        with open(agg_path, "w") as f:
            json.dump(agg, f, indent=2)
        print(f"\nUpdated {agg_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("FPS SUMMARY")
    print("=" * 60)
    for name, fps in sorted(results.items()):
        ms = 1000 / max(fps, 0.1)
        print(f"  {name:<50} {fps:>6.1f} FPS  ({ms:.1f} ms)")
    print("=" * 60)


if __name__ == "__main__":
    main()
