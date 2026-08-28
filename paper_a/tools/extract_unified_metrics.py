"""
Comprehensive Metric Aggregator: Extracts and unifies all official benchmark & ablation metrics
from downloaded artifact directories and run ledgers.
"""
from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
RESULTS_DIR = ROOT / "journal/results"
sys.stdout.reconfigure(encoding="utf-8")

def parse_metrics_from_folder(folder_p: Path) -> dict:
    res = {}
    
    # 1. Search for metrics.json or summary.json
    for json_file in folder_p.rglob("*.json"):
        if "metrics" in json_file.name or "summary" in json_file.name:
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    res.update(data)
                elif isinstance(data, list) and data:
                    res["records"] = data
            except Exception:
                pass

    # 2. Search for metrics.csv
    for csv_file in folder_p.rglob("metrics.csv"):
        try:
            lines = csv_file.read_text(encoding="utf-8").splitlines()
            if len(lines) > 1:
                header = [h.strip() for h in lines[0].split(",")]
                last_row = [r.strip() for r in lines[-1].split(",")]
                res["csv_last"] = dict(zip(header, last_row))
                # Find best row by mAP50 or AP50
                best_row = None
                best_val = -1.0
                for line in lines[1:]:
                    vals = [v.strip() for v in line.split(",")]
                    row_dict = dict(zip(header, vals))
                    val_50 = float(row_dict.get("mAP_50", row_dict.get("coco_AP50", row_dict.get("AP50", 0.0))))
                    if val_50 > best_val:
                        best_val = val_50
                        best_row = row_dict
                if best_row:
                    res["csv_best"] = best_row
        except Exception:
            pass

    # 3. Search for logs
    for log_file in folder_p.rglob("*.log"):
        try:
            text = log_file.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if "COCO AP     :" in line:
                    res["coco_AP"] = float(line.split(":")[-1])
                elif "COCO AP@50  :" in line:
                    res["coco_AP50"] = float(line.split(":")[-1])
                elif "COCO AP@75  :" in line:
                    res["coco_AP75"] = float(line.split(":")[-1])
                elif "COCO AP_small :" in line:
                    res["coco_APS"] = float(line.split(":")[-1])
                elif "COCO AR@100 :" in line:
                    res["coco_AR100"] = float(line.split(":")[-1])
                elif "mAP(scale)  :" in line:
                    res["mAP_scale"] = float(line.split(":")[-1])
                elif "AP_micro(" in line:
                    res["AP_micro"] = float(line.split(":")[-1])
                elif "AP_tiny (" in line:
                    res["AP_tiny"] = float(line.split(":")[-1])
                elif "Precision   :" in line:
                    res["Precision"] = float(line.split(":")[-1])
                elif "Recall      :" in line:
                    res["Recall"] = float(line.split(":")[-1])
        except Exception:
            pass

    return res


def main():
    print("=" * 110)
    print("📊 UNIFIED BENCHMARK & ABLATION METRICS EXTRACTION")
    print("=" * 110)

    runs = [
        ("Faster R-CNN Baseline (Standard)", RESULTS_DIR / "amongus1504_tod-aitod-baseline-s42-20260823"),
        ("Cascade R-CNN Baseline", RESULTS_DIR / "hngtrngtn_tod-aitod-cascade-s42-20260824"),
        ("RFLA (Gaussian Assignment)", RESULTS_DIR / "hngngnguynvn_tod-aitod-rfla-s42-20260823"),
        ("NWD (Wasserstein Distance)", RESULTS_DIR / "dipphmngc_tod-aitod-nwd-s42-20260823"),
        ("IGWD (Gaussian Wasserstein)", RESULTS_DIR / "hienquang06_tod-aitod-igwd-s42-20260823"),
        ("SA-ALW Full (Paper A)", ROOT / "runs/sa_alw_full__la_loss__seed42"),
        ("H-WIoU: Pure W2 (gamma=0)", RESULTS_DIR / "thyngluthy_tod-tp-ablation-pure-w2-s42"),
        ("H-WIoU: Pure IoU (gamma=1)", RESULTS_DIR / "hienquang06_tod-tp-ablation-pure-iou-s42"),
        ("H-WIoU: Static gamma=0.5", RESULTS_DIR / "dipphmngc_tod-tp-ablation-static-half-s42"),
        ("H-WIoU: Sigmoid Schedule", RESULTS_DIR / "pptlyn11_tod-tp-ablation-form-sigmoid-s42"),
        ("H-WIoU: Exponential Schedule", RESULTS_DIR / "hngngnguynvn_tod-tp-ablation-exp-form-s42"),
        ("H-WIoU: Pivot sigma_0=4px", RESULTS_DIR / "hngtrngtn_tod-tp-ablation-sig4-s42"),
        ("H-WIoU: Pivot sigma_0=12px", RESULTS_DIR / "luongsythanh_tod-tp-ablation-sig12-s42"),
        ("H-WIoU: LA Only", RESULTS_DIR / "dipphmngc_tod-tp-ablation-place-la-s42"),
        ("H-WIoU: Loss Only", RESULTS_DIR / "hienquang06_tod-tp-ablation-place-loss-s42"),
    ]

    for name, p in runs:
        if p.exists():
            data = parse_metrics_from_folder(p)
            print(f"\n[{name}] ({p.name})")
            if "csv_best" in data:
                cb = data["csv_best"]
                print(f"  Best CSV: Ep={cb.get('epoch')} | mAP50={cb.get('mAP_50', cb.get('coco_AP50'))} | AP={cb.get('coco_AP')} | AP75={cb.get('coco_AP75')} | AP_micro={cb.get('AP_micro')} | AP_tiny={cb.get('AP_tiny')}")
            if "coco_AP50" in data:
                print(f"  Log Metric: mAP50={data.get('coco_AP50')} | AP={data.get('coco_AP')} | AP75={data.get('coco_AP75')} | AP_micro={data.get('AP_micro')} | AP_tiny={data.get('AP_tiny')} | Scale={data.get('mAP_scale')}")
            if not data:
                print("  (Checking files...)")

if __name__ == "__main__":
    main()
