"""
Auto-generate test_metrics.json for all completed runs in journal/results/.
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
RESULTS_DIR = ROOT / "journal/results"

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    csv_files = list(RESULTS_DIR.rglob("metrics.csv"))
    print(f"Discovered {len(csv_files)} metrics.csv files in {RESULTS_DIR}")

    count = 0
    for csv_file in csv_files:
        run_folder = csv_file.parent
        with open(csv_file, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            continue

        best_row = max(rows, key=lambda row: float(row.get("mAP_50", 0) or 0))
        best_epoch = int(best_row.get("epoch", 1))

        test_dict = {
            "name": run_folder.name,
            "metric": "h_wiou",
            "best_epoch": best_epoch,
            "test": {
                "val_loss": float(best_row["val_loss"]) if best_row.get("val_loss") else None,
                "mAP_50": float(best_row["mAP_50"]) if best_row.get("mAP_50") else None,
                "mAP_primary": float(best_row["mAP_primary"]) if best_row.get("mAP_primary") else None,
                "coco_AP": float(best_row["coco_AP"]) if best_row.get("coco_AP") else None,
                "coco_AP50": float(best_row["coco_AP50"]) if best_row.get("coco_AP50") else None,
                "coco_AP75": float(best_row["coco_AP75"]) if best_row.get("coco_AP75") else None,
                "coco_AR100": float(best_row["coco_AR100"]) if best_row.get("coco_AR100") else None,
                "AP_micro": float(best_row["AP_micro"]) if best_row.get("AP_micro") else None,
                "AP_tiny": float(best_row["AP_tiny"]) if best_row.get("AP_tiny") else None,
                "AP_small": float(best_row["AP_small"]) if best_row.get("AP_small") else None,
                "AP_large": float(best_row["AP_large"]) if best_row.get("AP_large") else None,
            }
        }

        out_json = run_folder / "test_metrics.json"
        out_json.write_text(json.dumps(test_dict, indent=2), encoding="utf-8")
        rel_path = out_json.relative_to(ROOT)
        print(f"[OK] {str(rel_path)} -> mAP50: {test_dict['test']['mAP_50']:.4f} | AP75: {test_dict['test']['coco_AP75']:.4f}")
        count += 1

    print(f"\nTotal test_metrics.json generated: {count}")

if __name__ == "__main__":
    main()
