"""
Test-set evaluation — locked evaluation on data/test/ for all checkpoints.

Usage:
    python scripts/test_eval.py                     # evaluate all checkpoints
    python scripts/test_eval.py --metric alw_full    # single metric
    python scripts/test_eval.py --output results.json # custom output path

Reads: runs/*/best.pt (or last.pt if best missing)
Writes: runs/test_results.json (all runs), runs/<name>/test_metrics.json (per-run)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import (
    SEED, seed_all,
    BATCH_SIZE, NUM_WORKERS, DEVICE,
)
from common.dataset import YOLOTinyDataset, collate_fn, compute_reliability_threshold
from common.metrics import get_metric_fn, METRIC_DISPLAY_NAME, NEEDS_RELIABILITY
from common.model import build_model
from common.eval_utils import evaluate


RUNS_DIR = ROOT / "runs"


def list_checkpoints(runs_dir: Path) -> List[Dict]:
    """Discover all checkpoints in runs/."""
    entries = []
    SKIP_PREFIXES = ("yolo_",)
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if any(run_dir.name.startswith(p) for p in SKIP_PREFIXES):
            continue
        best_pt = run_dir / "best.pt"
        last_pt = run_dir / "last.pt"
        if not best_pt.exists() and not last_pt.exists():
            continue
        ckpt = best_pt if best_pt.exists() else last_pt
        entry = {"run_dir": str(run_dir), "checkpoint": str(ckpt),
                 "name": run_dir.name, "ckpt_type": "best" if best_pt.exists() else "last"}
        parts = run_dir.name.split("__")
        if run_dir.name.startswith("frcnn_standard"):
            entry["metric"] = "iou"
            entry["placement"] = parts[1]
            entry["seed"] = int(parts[2].replace("seed", ""))
        else:
            entry["metric"] = parts[0]
            entry["placement"] = parts[1]
            entry["seed"] = int(parts[2].replace("seed", ""))
        entries.append(entry)
    return entries


def build_test_loader() -> DataLoader:
    """Build DataLoader for test set (no augmentation, no copy-paste)."""
    td = ROOT / "data" / "test"
    ds = YOLOTinyDataset(
        img_dir=td / "images",
        lbl_dir=td / "labels",
        is_train=False,
    )
    return DataLoader(
        ds, batch_size=2, shuffle=False,
        num_workers=0,  # safe for Windows
        collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )


def evaluate_checkpoint(entry: Dict, test_loader: DataLoader) -> Dict:
    """Load model from checkpoint, evaluate on test set."""
    ckpt_path = Path(entry["checkpoint"])
    metric = entry["metric"]
    placement = entry["placement"]
    seed_val = entry["seed"]

    print(f"\n{'='*60}")
    print(f"  {entry['name']}")
    print(f"  Metric: {metric}  |  Placement: {placement}  |  Seed: {seed_val}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"{'='*60}")

    # Load checkpoint config
    try:
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"  ERROR loading checkpoint: {e}")
        return {"name": entry["name"], "error": str(e)}

    stored_config = ck.get("config", {})
    stored_metric = stored_config.get("metric", metric)
    stored_seed = stored_config.get("seed", seed_val)
    stored_placement = stored_config.get("placement", placement)

    # Use stored config if available, fallback to parsed values
    effective_metric = stored_metric if stored_metric else metric
    effective_placement = stored_placement if stored_placement else placement
    effective_seed = stored_seed if stored_seed else seed_val

    print(f"  Stored config: metric={effective_metric}, "
          f"placement={effective_placement}, seed={effective_seed}")

    # FRCNN standard baseline: no custom metric, standard torchvision IoU
    if effective_metric == "iou":
        metric_fn = None
        effective_placement = "everywhere"
    else:
        try:
            metric_fn = get_metric_fn(effective_metric)
        except ValueError as e:
            print(f"  ERROR: unknown metric '{effective_metric}': {e}")
            return {"name": entry["name"], "error": f"Unknown metric: {effective_metric}"}

    # Compute reliability_thr if needed
    reliability_thr = 16.0
    if effective_metric in NEEDS_RELIABILITY or effective_placement in ("la_loss_nms",):
        # For test eval, use the stored value or default
        reliability_thr = stored_config.get("reliability_thr", 16.0)

    # Build model
    try:
        model = build_model(
            metric_fn=metric_fn,
            placement=effective_placement,
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
        ).to(DEVICE)
    except Exception as e:
        print(f"  ERROR building model: {e}")
        return {"name": entry["name"], "error": f"Model build failed: {e}"}

    # Load weights
    try:
        model.load_state_dict(ck["model"])
    except Exception as e:
        print(f"  ERROR loading weights: {e}")
        return {"name": entry["name"], "error": f"Weight load failed: {e}"}

    # Evaluate
    model.eval()
    try:
        metrics = evaluate(model, test_loader, DEVICE, measure_fps_flag=False)
    except Exception as e:
        print(f"  ERROR during evaluation: {e}")
        return {"name": entry["name"], "error": f"Eval failed: {e}"}

    result = {
        "name": entry["name"],
        "metric": effective_metric,
        "display_name": METRIC_DISPLAY_NAME.get(effective_metric, effective_metric),
        "placement": effective_placement,
        "seed": effective_seed,
        "ckpt_type": entry["ckpt_type"],
        "ckpt_epoch": ck.get("epoch", "unknown"),
        "best_mAP50_val": round(
            ck.get("best_mAP50", ck.get("best_metric_value", 0.0)) or 0.0, 4),
        "test": metrics,
    }
    print(f"\n  Test mAP(scale): {metrics.get('mAP_primary', 0):.4f}")
    print(f"  Test mAP@50:     {metrics.get('mAP_50', 0):.4f}")
    print(f"  Test AP_micro:   {metrics.get('AP_micro', 0):.4f}")
    print(f"  Test AP_tiny:    {metrics.get('AP_tiny', 0):.4f}")
    print(f"  Test AP_small:   {metrics.get('AP_small', 0):.4f}")
    print(f"  Test AP_large:   {metrics.get('AP_large', 0):.4f}")

    return result


def print_summary(results: List[Dict]):
    """Print a formatted summary table of test results."""
    print("\n\n" + "=" * 110)
    print("TEST-SET EVALUATION SUMMARY")
    print("=" * 110)
    header = (f"{'Experiment':<40} {'mAP(scale)':>10} {'AP_micro':>9} "
              f"{'AP_tiny':>9} {'AP_small':>9} {'AP_large':>9} {'COCO mAP@50':>11}")
    print(header)
    print("-" * 110)

    # Sort: by metric name
    valid = [r for r in results if "test" in r]
    valid.sort(key=lambda r: r.get("metric", "zzz"))

    for r in valid:
        t = r.get("test", {})
        name = r["name"][:38]
        row = (f"{name:<40} "
               f"{t.get('mAP_primary', 0):>10.4f} "
               f"{t.get('AP_micro', 0):>9.4f} "
               f"{t.get('AP_tiny', 0):>9.4f} "
               f"{t.get('AP_small', 0):>9.4f} "
               f"{t.get('AP_large', 0):>9.4f} "
               f"{t.get('mAP_50', 0):>11.4f}")
        print(row)

    print("-" * 110)
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  {e['name']}: {e['error']}")
    print("=" * 110)


def main():
    parser = argparse.ArgumentParser(description="Test-set evaluation for all checkpoints")
    parser.add_argument("--metric", type=str, default=None,
                        help="Only evaluate a specific metric (e.g., alw_full)")
    parser.add_argument("--output", type=str, default="test_results.json",
                        help="Output JSON file in runs/")
    args = parser.parse_args()

    seed_all(SEED)
    print(f"Device: {DEVICE}")

    # Build test loader
    print("\nBuilding test DataLoader...")
    test_loader = build_test_loader()
    print(f"Test set: {len(test_loader.dataset)} tiles")

    # Discover checkpoints
    entries = list_checkpoints(RUNS_DIR)
    if args.metric:
        entries = [e for e in entries if e.get("metric") == args.metric]

    print(f"\nFound {len(entries)} checkpoints:")
    for e in entries:
        print(f"  - {e['name']} ({e['ckpt_type']})")

    # Evaluate each
    results = []
    for entry in entries:
        result = evaluate_checkpoint(entry, test_loader)
        results.append(result)

        # Save per-run result
        run_dir = Path(entry["run_dir"])
        per_run_json = run_dir / "test_metrics.json"
        with open(per_run_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {per_run_json}")

    # Save aggregate
    output_path = RUNS_DIR / args.output
    aggregate = {
        "test_set": {
            "images": 65,
            "tiles": len(test_loader.dataset),
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"\nAggregate results saved to: {output_path}")

    print_summary(results)


if __name__ == "__main__":
    main()
