"""
Real-Time Cluster Monitor and Safe Checkpoint Downloader for the Active AI-TOD-v2 Kaggle GPU Cluster.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"
DOWNLOAD_DIR = ROOT / "runs" / "aitod_kaggle_checkpoints"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

ACTIVE_CLUSTER_KERNELS = [
    # 1. Core Proposed & Ablation Methods
    {"account": "amongus1504", "slug": "tod-aitod-hwiou-sig8-s42-rot", "cred": CREDS_DIR / "kaggle.json", "tag": "hwiou_sig8_s42", "desc": "H-WIoU Proposed (sigma0=8.0px, S42)"},
    {"account": "dipphmngc", "slug": "tod-aitod-hwiou-sig6-s42-rot", "cred": CREDS_DIR / "kaggle (11).json", "tag": "hwiou_sig6_s42", "desc": "H-WIoU Ablation (sigma0=6.0px, S42)"},
    {"account": "thyngluthy", "slug": "tod-aitod-hwiou-sig10-s42-v2", "cred": CREDS_DIR / "kaggle (4).json", "tag": "hwiou_sig10_s42", "desc": "H-WIoU Ablation (sigma0=10.0px, S42)"},
    {"account": "phuc1806", "slug": "tod-aitod-hwiou-cascade-s42-v2", "cred": CREDS_DIR / "kaggle (12).json", "tag": "hwiou_cascade_s42", "desc": "H-WIoU + Cascade R-CNN (S42)"},
    {"account": "pptlyn11", "slug": "tod-aitod-hwiou-sig8-s2024-v2", "cred": CREDS_DIR / "kaggle (9).json", "tag": "hwiou_sig8_s2024", "desc": "H-WIoU Multi-Seed (sigma0=8.0px, S2024)"},

    # 2. SOTA Baselines
    {"account": "amongus1504", "slug": "tod-aitod-baseline-s42-v2", "cred": CREDS_DIR / "kaggle.json", "tag": "baseline_s42", "desc": "Faster R-CNN Baseline (Standard IoU, S42)"},
    {"account": "dipphmngc", "slug": "tod-aitod-nwd-s42-v2", "cred": CREDS_DIR / "kaggle (11).json", "tag": "nwd_s42", "desc": "NWD NeurIPS 2021 (S42)"},
    {"account": "thyngluthy", "slug": "tod-aitod-igwd-s42-rot", "cred": CREDS_DIR / "kaggle (4).json", "tag": "igwd_s42", "desc": "IGWD IEEE TMM 2022 (S42)"},
    {"account": "phuc1806", "slug": "tod-aitod-rfla-s42-rot", "cred": CREDS_DIR / "kaggle (12).json", "tag": "rfla_s42", "desc": "RFLA ECCV 2022 (S42)"},
    {"account": "trieuvo123", "slug": "tod-aitod-safit-s42-v2", "cred": CREDS_DIR / "kaggle (10).json", "tag": "safit_s42", "desc": "SAFit AAAI 2024 (S42)"},
    {"account": "ngquangnht", "slug": "tod-aitod-baseline-s123-v2", "cred": ROOT / ".runtime/kaggle/wp02/multi_account/cfg_ngquangnht/kaggle.json", "tag": "baseline_s123", "desc": "Faster R-CNN Baseline (Multi-Seed S123)"},
]


def check_status(item: dict) -> str:
    account = item["account"]
    slug = item["slug"]
    cred_file = item["cred"]
    ref = f"{account}/{slug}"

    with tempfile.TemporaryDirectory() as tmp_prof:
        shutil.copy(cred_file, Path(tmp_prof) / "kaggle.json")
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(tmp_prof)
        cmd = [sys.executable, "-m", "kaggle", "kernels", "status", ref]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        out = res.stdout.strip()
        if "complete" in out.lower():
            return "COMPLETE"
        elif "running" in out.lower():
            return "RUNNING"
        elif "error" in out.lower():
            return "ERROR"
        elif "queued" in out.lower():
            return "QUEUED"
        return out if out else "UNKNOWN"


def download_output(item: dict) -> bool:
    account = item["account"]
    slug = item["slug"]
    cred_file = item["cred"]
    tag = item["tag"]
    ref = f"{account}/{slug}"
    target_dir = DOWNLOAD_DIR / tag
    target_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_prof:
        shutil.copy(cred_file, Path(tmp_prof) / "kaggle.json")
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(tmp_prof)
        cmd = [sys.executable, "-m", "kaggle", "kernels", "output", ref, "-p", str(target_dir)]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        print(f"  [DOWNLOAD] {ref} -> {target_dir}: {res.stdout.strip()}")
        return True


def status_snapshot():
    print("=" * 110)
    print("                      ACTIVE AI-TOD-v2 GPU CLUSTER STATUS SNAPSHOT                      ")
    print("=" * 110)
    print(f"{'No.':<4} | {'Method / Description':<40} | {'Kernel Reference':<36} | {'Status':<15}")
    print("-" * 110)

    running_count = 0
    complete_count = 0
    for idx, item in enumerate(ACTIVE_CLUSTER_KERNELS, 1):
        ref = f"{item['account']}/{item['slug']}"
        st = check_status(item)
        if st in ["RUNNING", "QUEUED"]:
            running_count += 1
        elif st == "COMPLETE":
            complete_count += 1
        print(f"{idx:<4} | {item['desc']:<40} | {ref:<36} | {st:<15}")

    print("=" * 110)
    print(f"Total Active Tasks: {len(ACTIVE_CLUSTER_KERNELS)} | Running: {running_count} | Completed: {complete_count}")
    print("=" * 110)


if __name__ == "__main__":
    status_snapshot()
