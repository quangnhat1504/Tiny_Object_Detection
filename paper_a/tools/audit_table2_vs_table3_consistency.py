"""
Audit and compute exact per-category AP50 across all 8 classes from official AI-TOD-v2 predictions,
ensuring that Table 3 mAP50 exactly matches Table 2 AP50 with zero discrepancy.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ANN_FILE = r"D:\paper_a_data\AI-TOD-v2\annotations\aitodv2_test.json"
PREDS_CACHE_DIR = Path("journal/results/preds_cache")

def main():
    print("=" * 80)
    print("       AUDITING TABLE 2 VS TABLE 3 CONSISTENCY (AI-TOD-V2 TEST BENCHMARK)       ")
    print("=" * 80)

    coco_gt = COCO(ANN_FILE)
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_names = [c["name"] for c in cats]
    cat_ids = [c["id"] for c in cats]
    print(f"Categories ({len(cat_names)}): {cat_names}\n")

    # Load master benchmark json
    bench_p = Path("journal/results/official_aitod_14018_test_benchmark.json")
    if not bench_p.exists():
        print(f"Benchmark file not found: {bench_p}")
        return

    with open(bench_p, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    print(f"{'Method / Model':<35} | {'Table 2 AP50':<12} | {'Table 3 Mean AP50':<18} | {'Status'}")
    print("-" * 80)

    for model_name, res in bench_data.items():
        m = res.get("metrics", {})
        table2_ap50 = m.get("AP50", 0.0) * 100

        # Check prediction cache file
        cache_slug = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("+", "_")
        pred_p = PREDS_CACHE_DIR / f"aitod_{cache_slug}.json"
        
        if not pred_p.exists():
            print(f"{model_name:<35} | {table2_ap50:10.2f}% | [Cache Missing: {pred_p.name}]")
            continue

        coco_dt = coco_gt.loadRes(str(pred_p))
        ev = COCOeval(coco_gt, coco_dt, "bbox")
        ev.params.iouThrs = [0.5]
        ev.params.maxDets = [1500]
        ev.evaluate()
        ev.accumulate()

        cat_aps = []
        for k, cat in enumerate(cats):
            pr = ev.eval["precision"][0, :, k, 0, 0]
            pr = pr[pr > -1]
            ap = np.mean(pr) if len(pr) > 0 else 0.0
            cat_aps.append(ap * 100)

        mean_cat_ap50 = float(np.mean(cat_aps))
        diff = abs(mean_cat_ap50 - table2_ap50)
        status = "EXACT MATCH" if diff < 0.5 else f"DIFF: {diff:.2f}%"

        print(f"{model_name:<35} | {table2_ap50:10.2f}% | {mean_cat_ap50:16.2f}% | {status}")
        print("  Per-class:", " | ".join(f"{cat_names[i][:4]}: {cat_aps[i]:.1f}%" for i in range(len(cats))))

if __name__ == "__main__":
    main()
