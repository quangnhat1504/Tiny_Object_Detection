"""
Real-time Polling, Progress Tracker, and Artifact Downloader for Official TinyPerson Kaggle Runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CREDS = Path.home() / ".kaggle"
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_official_tp_profiles")
DOWNLOAD_DIR = ROOT / "runs" / "official_tinyperson_kaggle_downloaded"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

EXPERIMENTS = [
    {
        "method": "standard",
        "tag": "tp_official_standard_s42",
        "slug": "tinyperson-official-standard-baseline-s42",
        "account": "amongus1504",
        "cred": "kaggle (2).json",
    },
    {
        "method": "nwd",
        "tag": "tp_official_nwd_s42",
        "slug": "tinyperson-official-nwd-s42",
        "account": "qnhat1504",
        "cred": "kaggle (3).json",
    },
    {
        "method": "igwd",
        "tag": "tp_official_igwd_s42",
        "slug": "tinyperson-official-igwd-s42",
        "account": "thyngluthy",
        "cred": "kaggle (4).json",
    },
    {
        "method": "sa_alw_full",
        "tag": "tp_official_saalw_s42",
        "slug": "tinyperson-official-sa-alw-s42",
        "account": "quangnhtng",
        "cred": "kaggle (6).json",
    },
    {
        "method": "h_wiou_sig8",
        "tag": "tp_official_hwiou_sig8_s42",
        "slug": "tinyperson-official-h-wiou-sigma8-s42",
        "account": "hienquang06",
        "cred": "kaggle (5).json",
    },
    {
        "method": "h_wiou_sig6",
        "tag": "tp_official_hwiou_sig6_s42",
        "slug": "tinyperson-official-h-wiou-sigma6-s42",
        "account": "hngngnguynvn",
        "cred": "kaggle (1).json",
    },
    {
        "method": "rfla",
        "tag": "tp_official_rfla_s42",
        "slug": "tinyperson-official-rfla-s42",
        "account": "luongsythanh",
        "cred": "kaggle (8).json",
    },
]

def check_kernel_status(exp: dict) -> tuple[str, str]:
    account = exp["account"]
    slug = exp["slug"]
    cred_file = exp["cred"]
    profile = PROFILE_ROOT / account

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(profile)

    cmd = [sys.executable, "-m", "kaggle", "kernels", "status", f"{account}/{slug}"]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    status_str = res.stdout.strip()
    return status_str, res.stderr.strip()

def download_kernel_output(exp: dict) -> bool:
    account = exp["account"]
    slug = exp["slug"]
    tag = exp["tag"]
    profile = PROFILE_ROOT / account
    target_dir = DOWNLOAD_DIR / tag
    target_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(profile)

    print(f"Downloading artifacts for {account}/{slug} into {target_dir}...")
    cmd = [sys.executable, "-m", "kaggle", "kernels", "output", f"{account}/{slug}", "-p", str(target_dir)]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("Download Output:", res.stdout.strip())
    return (target_dir / "runs").exists() or any(target_dir.rglob("best*.pt"))

def main():
    print("=== MONITORING 7 OFFICIAL TINYPERSON KAGGLE GPU RUNS ===")
    completed = set()

    for exp in EXPERIMENTS:
        status_text, err = check_kernel_status(exp)
        print(f"[{exp['account']:<14}] {exp['tag']:<30} -> Status: {status_text}")

if __name__ == "__main__":
    main()
