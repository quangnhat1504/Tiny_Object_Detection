"""
Inspect downloaded artifacts, logs, metrics, and checkpoints across all completed runs.
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

TARGET_DIRS = [
    ("amongus1504_tod-aitod-baseline-s42-20260823", "Faster R-CNN Baseline"),
    ("hngtrngtn_tod-aitod-cascade-s42-20260824", "Cascade R-CNN Baseline"),
    ("hngngnguynvn_tod-aitod-rfla-s42-20260823", "RFLA (Gaussian Assign)"),
    ("dipphmngc_tod-aitod-nwd-s42-20260823", "NWD (Wasserstein Loss)"),
    ("hienquang06_tod-aitod-igwd-s42-20260823", "IGWD (Gaussian Wasserstein)"),
]


def inspect_run(folder_name: str, label: str):
    folder = RESULTS_DIR / folder_name
    print("=" * 80)
    print(f"🔍 Inspecting: {label} ({folder_name})")
    print("=" * 80)

    if not folder.exists():
        print(f"❌ Folder not found: {folder}")
        return

    # 1. List all files
    all_files = list(folder.rglob("*"))
    files = [f for f in all_files if f.is_file()]
    print(f"Total files: {len(files)}")
    for f in files:
        rel = f.relative_to(folder)
        size = f.stat().st_size
        print(f"  - {rel} ({size:,} bytes)")

    # 2. Check for logs and extract metrics
    logs = list(folder.rglob("*.log")) + list(folder.rglob("*.txt"))
    for log_p in logs:
        if log_p.stat().st_size == 0:
            continue
        try:
            content = log_p.read_text(encoding="utf-8", errors="replace")
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            metric_lines = [
                l for l in lines 
                if any(k in l for k in ["mAP", "AP50", "AP_vt", "AP_t", "AP_s", "AP_m", "AP_primary", "coco_AP", "Best AP", "Evaluating Epoch"])
            ]
            print(f"\n  [Log Snippets from {log_p.name}]:")
            if metric_lines:
                for ml in metric_lines[-8:]:
                    print(f"    > {ml}")
            else:
                for tl in lines[-5:]:
                    print(f"    > {tl}")
        except Exception as e:
            print(f"  Error reading log {log_p}: {e}")

    # 3. Check for metrics.json or metrics.csv
    metric_files = list(folder.rglob("metrics.json")) + list(folder.rglob("metrics.csv")) + list(folder.rglob("summary.json"))
    for mf in metric_files:
        if mf.stat().st_size > 0:
            print(f"\n  [Structured Metric File: {mf.name}]:")
            try:
                text = mf.read_text(encoding="utf-8", errors="replace")
                print(text[:500])
            except Exception as e:
                print(f"  Error reading {mf}: {e}")
    print()


def main():
    for f_name, label in TARGET_DIRS:
        inspect_run(f_name, label)


if __name__ == "__main__":
    main()
