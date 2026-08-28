"""
Unified Real-Time Log Monitor & Multi-Metric Vector Streamer for 6 Active Kaggle GPU T4 Runs.
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

RUNNING_KERNELS = [
    {
        "id": "aitod_baseline",
        "account": "amongus1504",
        "ref": "amongus1504/tod-aitod-baseline-s42-20260820",
        "dataset": "AI-TOD-v2",
        "name": "Faster R-CNN Baseline (AI-TOD-v2)",
    },
    {
        "id": "aitod_hwiou_sig8",
        "account": "quangnhtng",
        "ref": "quangnhtng/tod-aitod-hwiou-sig8-s42-20260820",
        "dataset": "AI-TOD-v2",
        "name": "H-WIoU sigma=8.0px (AI-TOD-v2)",
    },
    {
        "id": "aitod_hwiou_sig6",
        "account": "qnhat1504",
        "ref": "qnhat1504/tod-aitod-hwiou-sig6-s42-20260820",
        "dataset": "AI-TOD-v2",
        "name": "H-WIoU sigma=6.0px (AI-TOD-v2)",
    },
    {
        "id": "tp_ablation_pure_w2",
        "account": "thyngluthy",
        "ref": "thyngluthy/tod-tp-ablation-pure-w2-s42-20260820",
        "dataset": "TinyPerson",
        "name": "TinyPerson Ablation: Pure W2 (gamma=0)",
    },
    {
        "id": "tp_ablation_pure_iou",
        "account": "hngngnguynvn",
        "ref": "hngngnguynvn/tod-tp-ablation-pure-iou-s42-20260820",
        "dataset": "TinyPerson",
        "name": "TinyPerson Ablation: Pure IoU (gamma=1)",
    },
    {
        "id": "tp_ablation_static_half",
        "account": "dipphmngc",
        "ref": "dipphmngc/tod-tp-ablation-static-half-s42-20260820",
        "dataset": "TinyPerson",
        "name": "TinyPerson Ablation: Static gamma=0.5",
    },
]


def check_kernel(k: dict):
    account = k["account"]
    ref = k["ref"]
    name = k["name"]
    ds = k["dataset"]
    profile = PROFILE_ROOT / account

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(profile)

    status_cmd = [sys.executable, "-m", "kaggle", "kernels", "status", ref]
    s_res = subprocess.run(status_cmd, env=env, capture_output=True, text=True)
    status_text = s_res.stdout.strip()
    if not status_text and s_res.stderr.strip():
        status_text = s_res.stderr.strip()

    print(f"\n[{ds}] {name} ({account}) -> {status_text}")

    # Fetch logs
    runner_cmd = [
        sys.executable, "-c",
        f"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
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
    log_res = subprocess.run(runner_cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    log_text = log_res.stdout

    # Save log
    log_file = RESULTS_DIR / f"{k['id']}_raw.log"
    log_file.write_text(log_text, encoding="utf-8")

    # If complete, download output directory
    if "COMPLETE" in status_text:
        out_dir = RESULTS_DIR / f"{account}_{ref.split('/')[-1]}"
        out_dir.mkdir(parents=True, exist_ok=True)
        download_cmd = [
            sys.executable, "-c",
            f"""
import os, sys
sys.stdout.reconfigure(encoding='utf-8')
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
try:
    api.kernels_output('{ref}', path=r'{out_dir}')
    print('DOWNLOAD_SUCCESS')
except Exception as e:
    print('DOWNLOAD_ERROR:', e)
"""
        ]
        d_res = subprocess.run(download_cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if "DOWNLOAD_SUCCESS" in d_res.stdout:
            print(f"  --> Downloaded full kernel artifacts to {out_dir.name}")

    lines = [l.strip() for l in log_text.splitlines() if l.strip()]
    latest_metrics = [l for l in lines if ("Epoch " in l and ("AP=" in l or "mAP@50" in l or "AP50=" in l))]
    if latest_metrics:
        print(f"  --> Latest Metric: {latest_metrics[-1]}")
    elif len(lines) > 0:
        print(f"  --> Tail Log: {lines[-1][:120]}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 105)
    print("                ACTIVE KAGGLE CLOUD GPU T4 BENCHMARK & ABLATION CLUSTER MONITOR                ")
    print("=" * 105)

    for k in RUNNING_KERNELS:
        check_kernel(k)

    print("\n" + "=" * 105)


if __name__ == "__main__":
    main()
