"""
Recompute full predictions and per-class AP50 for Baseline, NWD, SAFit, and H-WIoU
to guarantee 100% mathematical unity between Table 2 (Overall) and Table 3 (Per-Class).
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from common.model import build_model
from common.metrics import get_metric_fn, configure_metric
from paper_a.evaluation.aitodv2_official import AITODv2Dataset, aitod_collate_fn, evaluate_aitod_model

ANN_FILE = Path(r"D:\paper_a_data\AI-TOD-v2\annotations\aitodv2_test.json")
IMG_DIR = Path(r"D:\paper_a_data\AI-TOD-v2\AI-TOD\images\test")
CACHE_DIR = Path("journal/results/preds_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print(f"       RECOMPUTING FULL PREDICTIONS FOR TABLE 2 & TABLE 3 UNITY (GPU: {torch.cuda.get_device_name(0)})       ")
    print("=" * 80)

    coco_gt = COCO(str(ANN_FILE))
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_names = [c["name"] for c in cats]

    ds = AITODv2Dataset(IMG_DIR, ANN_FILE, drop_empty=False)
    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=aitod_collate_fn)

    # Models to check and evaluate
    models = [
        (
            "Faster R-CNN Baseline",
            ROOT / "runs/aitod_kaggle_checkpoints/baseline_s42/tod_output/runs/aitod_standard_everywhere_smooth_l1_s42_baseline_s42/best.pt",
            None,
            "everywhere",
            None,
        ),
        (
            "NWD (NeurIPS 2021)",
            ROOT / "runs/aitod_kaggle_checkpoints/nwd_s42/tod_output/runs/aitod_nwd_everywhere_metric_s42_nwd_s42/best.pt",
            "nwd",
            "everywhere",
            None,
        ),
        (
            "SAFit (AAAI 2024)",
            ROOT / "runs/official_aitod_checkpoints/safit/aitod_safit_s42/runs/aitod_sa_alw_canonical_everywhere_metric_s42_safit/last.pt",
            "sa_alw_canonical",
            "everywhere",
            None,
        ),
    ]

    all_table3_rows = {}

    for name, ckpt_p, m_name, placement, sigma0 in models:
        if not ckpt_p.exists():
            print(f"[SKIP] Checkpoint missing: {ckpt_p}")
            continue

        print("\n" + "#" * 80)
        print(f"--> EVALUATING FULL DENSITY: {name}")
        print("#" * 80)

        if m_name == "h_wiou":
            m_fn, _, _ = configure_metric("h_wiou", h_wiou_sigma_0=sigma0 if sigma0 else 8.0)
        elif m_name:
            m_fn = get_metric_fn(m_name)
        else:
            m_fn = None

        model = build_model(
            num_classes=9,
            metric_fn=m_fn,
            placement=placement,
            box_loss_type="h_wiou" if m_name == "h_wiou" else ("metric" if m_name else "smooth_l1"),
            box_loss_warmup_epochs=0,
        ).to(device)

        ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
        sd = ck.get("model", ck.get("model_state_dict", ck.get("state_dict", ck)))
        model.load_state_dict(sd, strict=False)

        cache_slug = name.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "").replace("+", "_")
        cache_p = CACHE_DIR / f"aitod_full_{cache_slug}.json"

        metrics = evaluate_aitod_model(
            model, loader, device, ANN_FILE,
            label_to_category_id=ds.label_to_category_id,
            cache_path=cache_p
        )

        # Compute per-category
        coco_dt = coco_gt.loadRes(str(cache_p))
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
            cat_aps.append(float(ap * 100))

        all_table3_rows[name] = {
            "per_class": cat_aps,
            "mean_ap50": float(np.mean(cat_aps)),
            "table2_ap50": float(metrics.get("AP50", 0.0) * 100),
            "table2_ap": float(metrics.get("AP", 0.0) * 100),
        }

        print(f"Results for {name}:")
        print(f"  Table 2 AP50: {metrics.get('AP50', 0.0)*100:.2f}% | Table 3 Mean: {np.mean(cat_aps):.2f}%")
        print("  Per-class:", " | ".join(f"{cat_names[i]}: {cat_aps[i]:.1f}%" for i in range(len(cats))))

    # Save to JSON
    out_json = Path("journal/results/table2_table3_unified_benchmark.json")
    out_json.write_text(json.dumps(all_table3_rows, indent=2), encoding="utf-8")
    print(f"\n[SUCCESS] Unified benchmark saved to {out_json.resolve()}")

if __name__ == "__main__":
    main()
