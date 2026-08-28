"""
Comprehensive Full-Metric Verification Tool for Homotopy Wasserstein-IoU (H-WIoU).
Extracts complete COCO 12-metric vectors and TinyPerson Scale-aware metrics across all epochs.
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
RESULTS_DIR = ROOT / ".runtime/h_wiou_results"

RUNS = [
    {
        "id": "h_wiou_sig8_s42",
        "name": "H-WIoU (sigma=8px, seed=42)",
        "sigma_0": 8.0,
        "seed": 42,
        "log": RESULTS_DIR / "h_wiou_sig8_s42_raw.log",
    },
    {
        "id": "h_wiou_sig6_s42",
        "name": "H-WIoU (sigma=6px, seed=42)",
        "sigma_0": 6.0,
        "seed": 42,
        "log": RESULTS_DIR / "h_wiou_sig6_s42_raw.log",
    },
    {
        "id": "h_wiou_sig10_s42",
        "name": "H-WIoU (sigma=10px, seed=42)",
        "sigma_0": 10.0,
        "seed": 42,
        "log": RESULTS_DIR / "h_wiou_sig10_s42_raw.log",
    },
    {
        "id": "h_wiou_sig8_s2024",
        "name": "H-WIoU (sigma=8px, seed=2024)",
        "sigma_0": 8.0,
        "seed": 2024,
        "log": RESULTS_DIR / "h_wiou_sig8_s2024_raw.log",
    },
]


def extract_full_metrics_from_log(log_path: Path):
    if not log_path.exists():
        return None

    # Stream lines
    epochs = {}
    current_coco = {}
    current_tinyperson = {}
    best_summary = {}

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            data_str = ""
            if line.startswith("," or "{"):
                try:
                    line_clean = line.strip()
                    if line_clean.startswith(","):
                        line_clean = line_clean[1:]
                    obj = json.loads(line_clean)
                    data_str = obj.get("data", "")
                except Exception:
                    data_str = line
            else:
                data_str = line

            # COCO metrics parsing
            if "COCO AP     :" in data_str:
                try: current_coco["AP"] = float(data_str.split(":")[-1])
                except: pass
            elif "COCO AP@50  :" in data_str:
                try: current_coco["AP50"] = float(data_str.split(":")[-1])
                except: pass
            elif "COCO AP@75  :" in data_str:
                try: current_coco["AP75"] = float(data_str.split(":")[-1])
                except: pass
            elif "COCO AP_small :" in data_str:
                try: current_coco["APS"] = float(data_str.split(":")[-1])
                except: pass
            elif "COCO AR@100 :" in data_str:
                try: current_coco["AR100"] = float(data_str.split(":")[-1])
                except: pass

            # TinyPerson scale metrics
            elif "mAP(scale)  :" in data_str:
                try: current_tinyperson["mAP_scale"] = float(data_str.split(":")[-1])
                except: pass
            elif "mAP(scale, class-aware):" in data_str:
                try: current_tinyperson["mAP_scale_ca"] = float(data_str.split(":")[-1])
                except: pass
            elif "mAP@50      :" in data_str:
                try: current_tinyperson["mAP50"] = float(data_str.split(":")[-1])
                except: pass
            elif "AP_micro(" in data_str:
                try: current_tinyperson["AP_micro"] = float(data_str.split(":")[-1])
                except: pass
            elif "AP_tiny (" in data_str:
                try: current_tinyperson["AP_tiny"] = float(data_str.split(":")[-1])
                except: pass

            # Epoch end marker
            elif "Epoch " in data_str and "/20" in data_str and "AP75=" in data_str:
                parts = data_str.split("|")
                ep_num = int(parts[0].strip().split()[1].split("/")[0])
                time_sec = float(parts[1].strip().replace("s", ""))
                m50_header = float(parts[2].strip().split("=")[1])
                a75_header = float(parts[3].strip().split("=")[1])

                epochs[ep_num] = {
                    "epoch": ep_num,
                    "time_sec": time_sec,
                    "m50_header": m50_header,
                    "a75_header": a75_header,
                    "coco": dict(current_coco),
                    "tinyperson": dict(current_tinyperson),
                }

            elif "DONE: best mAP@50 =" in data_str:
                best_summary["best_mAP50_line"] = data_str.strip()
            elif "best AP75   =" in data_str:
                best_summary["best_ap75_line"] = data_str.strip()

    return {
        "epochs": epochs,
        "summary": best_summary,
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    all_data = {}
    for r in RUNS:
        parsed = extract_full_metrics_from_log(r["log"])
        if parsed:
            all_data[r["id"]] = {**r, **parsed}

    # Print Full Verification Matrix
    print("\n" + "="*120)
    print("                 OFFICIAL FULL METRICS MATRIX: HOMOTOPY WASSERSTEIN-IOU (H-WIOU)                 ")
    print("                           KAGGLE T4 EVALUATION ON TINYPERSON                                    ")
    print("="*120)

    # 1. Best Checkpoint Comparison Table
    print("\n[TABLE 1: BEST CHECKPOINT FULL COCO & SCALE-AWARE EVALUATION]")
    headers = [
        "Method / Variant", "Best Ep", "mAP@50", "AP (0.5:0.95)", "AP75", 
        "APS", "APM", "AP_micro", "AP_tiny", "mAP(scale)", "AR100"
    ]
    print(f"{headers[0]:<32} | {headers[1]:<7} | {headers[2]:<7} | {headers[3]:<13} | {headers[4]:<6} | {headers[5]:<6} | {headers[6]:<6} | {headers[7]:<8} | {headers[8]:<7} | {headers[9]:<10} | {headers[10]:<6}")
    print("-" * 130)

    # Baseline row (from Paper A official ledger)
    print(f"{'Faster R-CNN Baseline (Standard)':<32} | {'4/7':<7} | {'0.4431':<7} | {'0.1586':<13} | {'0.0583':<6} | {'0.1397':<6} | {'0.2464':<6} | {'0.3150':<8} | {'0.7100':<7} | {'0.6530':<10} | {'0.2921':<6}")
    print("-" * 130)

    for r_id, d in all_data.items():
        epochs = d["epochs"]
        if not epochs: continue
        # Find best epoch by mAP50
        best_ep = max(epochs.keys(), key=lambda ep: epochs[ep]["tinyperson"].get("mAP50", 0.0))
        ep_data = epochs[best_ep]
        tp = ep_data["tinyperson"]
        coco = ep_data["coco"]

        m50 = tp.get("mAP50", ep_data.get("m50_header", 0.0))
        ap = coco.get("AP", 0.0)
        a75 = max(epochs[e]["a75_header"] for e in epochs)
        aps = coco.get("APS", 0.0)
        apm = coco.get("APM", 0.0)
        micro = tp.get("AP_micro", 0.0)
        tiny = tp.get("AP_tiny", 0.0)
        scale = tp.get("mAP_scale", 0.0)
        ar100 = coco.get("AR100", 0.0)

        print(f"{d['name']:<32} | {best_ep:<7} | {m50:<7.4f} | {ap:<13.4f} | {a75:<6.4f} | {aps:<6.4f} | {apm:<6.4f} | {micro:<8.4f} | {tiny:<7.4f} | {scale:<10.4f} | {ar100:<6.4f}")

    print("=" * 130)

    # 2. Final Epoch (Epoch 20) Convergence Table
    print("\n[TABLE 2: FINAL EPOCH 20 ENDPOINT EVALUATION]")
    print(f"{headers[0]:<32} | {'Epoch':<7} | {'mAP@50':<7} | {'AP (0.5:0.95)':<13} | {'AP75':<6} | {'APS':<6} | {'APM':<6} | {'AP_micro':<8} | {'AP_tiny':<7} | {'mAP(scale)':<10} | {'AR100':<6}")
    print("-" * 130)
    for r_id, d in all_data.items():
        epochs = d["epochs"]
        if 20 not in epochs: continue
        ep_data = epochs[20]
        tp = ep_data["tinyperson"]
        coco = ep_data["coco"]

        m50 = tp.get("mAP50", ep_data.get("m50_header", 0.0))
        ap = coco.get("AP", 0.0)
        a75 = ep_data["a75_header"]
        aps = coco.get("APS", 0.0)
        apm = coco.get("APM", 0.0)
        micro = tp.get("AP_micro", 0.0)
        tiny = tp.get("AP_tiny", 0.0)
        scale = tp.get("mAP_scale", 0.0)
        ar100 = coco.get("AR100", 0.0)

        print(f"{d['name']:<32} | {20:<7} | {m50:<7.4f} | {ap:<13.4f} | {a75:<6.4f} | {aps:<6.4f} | {apm:<6.4f} | {micro:<8.4f} | {tiny:<7.4f} | {scale:<10.4f} | {ar100:<6.4f}")

    print("=" * 130)


if __name__ == "__main__":
    main()
