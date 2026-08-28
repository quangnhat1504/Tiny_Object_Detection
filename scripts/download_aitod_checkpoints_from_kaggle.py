"""
Download all trained AI-TOD-v2 checkpoints from Kaggle cluster into runs/aitod_kaggle_checkpoints/
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CREDS_DIR = Path.home() / ".kaggle"
DOWNLOAD_DIR = ROOT / "runs" / "aitod_kaggle_checkpoints"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_aitod_profiles")
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)

AITOD_RUNS = [
    {"account": "amongus1504", "slug": "tod-aitod-baseline-s42", "cred": "kaggle (2).json", "tag": "baseline"},
    {"account": "qnhat1504", "slug": "tod-aitod-hwiou-sig8-s42", "cred": "kaggle (3).json", "tag": "hwiou_sig8"},
    {"account": "quangnhtng", "slug": "tod-aitod-hwiou-sig6-s42", "cred": "kaggle (6).json", "tag": "hwiou_sig6"},
    {"account": "dipphmngc", "slug": "tod-aitod-nwd-s42", "cred": "kaggle (11).json", "tag": "nwd"},
    {"account": "hienquang06", "slug": "tod-aitod-igwd-s42", "cred": "kaggle (5).json", "tag": "igwd"},
    {"account": "hngngnguynvn", "slug": "tod-aitod-rfla-s42", "cred": "kaggle (1).json", "tag": "rfla"},
    {"account": "phuc1806", "slug": "tod-aitod-hwiou-cascade-s42", "cred": "kaggle (12).json", "tag": "hwiou_cascade"},
]

def download_run(run: dict):
    account = run["account"]
    slug = run["slug"]
    tag = run["tag"]
    cred_file = run["cred"]

    profile = PROFILE_ROOT / account
    profile.mkdir(parents=True, exist_ok=True)
    shutil.copy(CREDS_DIR / cred_file, profile / "kaggle.json")

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(profile)

    target_dir = DOWNLOAD_DIR / tag
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading checkpoint from {account}/{slug} into {target_dir}...")
    cmd = [sys.executable, "-m", "kaggle", "kernels", "output", f"{account}/{slug}", "-p", str(target_dir)]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("Output:", res.stdout.strip())
    if res.stderr.strip():
        print("Stderr:", res.stderr.strip())

def main():
    print("=== DOWNLOADING AI-TOD-V2 CHECKPOINTS FROM KAGGLE ===")
    for run in AITOD_RUNS:
        download_run(run)
    print("\n=== CHECKING DOWNLOADED CHECKPOINTS ===")
    for ck in DOWNLOAD_DIR.rglob("best*.pt"):
        print(" ", ck.relative_to(DOWNLOAD_DIR), f"({ck.stat().st_size / 1e6:.1f} MB)")

if __name__ == "__main__":
    main()
