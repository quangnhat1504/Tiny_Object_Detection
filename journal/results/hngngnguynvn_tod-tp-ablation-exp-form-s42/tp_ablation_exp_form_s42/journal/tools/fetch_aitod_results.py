"""
Fast Streaming Log Fetcher & Multi-Metric Extractor for AI-TOD-v2 & Journal Ablations.
Uses isolated sub-processes to cleanly switch Kaggle account credentials without API cache collisions.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_hwiou_profiles")
RESULTS_DIR = ROOT / "journal/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

KERNELS = [
    {
        "id": "aitod_baseline_s42",
        "account": "amongus1504",
        "ref": "amongus1504/tod-ai-tod-baseline-s42-20260820",
        "type": "aitod",
        "name": "Faster R-CNN Baseline (AI-TOD-v2)",
    },
    {
        "id": "aitod_hwiou_sig8_s42",
        "account": "qnhat1504",
        "ref": "qnhat1504/tod-ai-tod-h-wiou-sigma8-s42-20260820",
        "type": "aitod",
        "name": "H-WIoU (sigma=8px, AI-TOD-v2)",
    },
    {
        "id": "aitod_hwiou_sig6_s42",
        "account": "quangnhtng",
        "ref": "quangnhtng/tod-ai-tod-h-wiou-sigma6-s42-20260820",
        "type": "aitod",
        "name": "H-WIoU (sigma=6px, AI-TOD-v2)",
    },
    {
        "id": "tp_ablation_pure_w2",
        "account": "thyngluthy",
        "ref": "thyngluthy/tod-tp-ablation-pure-w2-s42-20260820",
        "type": "tinyperson",
        "name": "TinyPerson Ablation Pure W2 (gamma=0)",
    },
]


def check_single_kernel(k: dict):
    account = k["account"]
    ref = k["ref"]
    name = k["name"]
    profile = PROFILE_ROOT / account

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(profile)

    # 1. Get status
    status_cmd = [sys.executable, "-m", "kaggle", "kernels", "status", ref]
    s_res = subprocess.run(status_cmd, env=env, capture_output=True, text=True)
    status_text = s_res.stdout.strip()
    if not status_text and s_res.stderr.strip():
        status_text = s_res.stderr.strip()

    print(f"\n[{name}] ({account}) -> {status_text}")

    # 2. Get logs
    log_cmd = [sys.executable, "-m", "kaggle", "kernels", "output", ref]
    # We can also get output stream
    runner_cmd = [
        sys.executable, "-c",
        f"""
import os, sys
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
try:
    res = api.kernels_logs('{ref}')
    log = res.get('log', '') if isinstance(res, dict) else str(res)
    print(log)
except Exception as e:
    print('ERROR:', e)
"""
    ]
    log_res = subprocess.run(runner_cmd, env=env, capture_output=True, text=True)
    log_text = log_res.stdout

    log_file = RESULTS_DIR / f"{k['id']}_raw.log"
    log_file.write_text(log_text, encoding="utf-8")

    lines = [l.strip() for l in log_text.splitlines() if l.strip()]
    latest_metrics = [l for l in lines if ("Epoch " in l and ("AP=" in l or "mAP@50" in l))]
    if latest_metrics:
        print(f"  --> Latest Metric: {latest_metrics[-1]}")
    elif len(lines) > 0:
        print(f"  --> Tail Log: {lines[-1][:120]}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 80)
    print("   AI-TOD-v2 & JOURNAL ABLATION STATUS & LOG MONITOR (KAGGLE CLOUD T4)   ")
    print("=" * 80)

    for k in KERNELS:
        check_single_kernel(k)

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
