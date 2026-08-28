"""
Download official AI-TOD-v2 completed model outputs from Kaggle into runs/official_aitod_checkpoints/
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
DOWNLOAD_DIR = ROOT / "runs" / "official_aitod_checkpoints"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_aitod_fetch_profiles")
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)

COMPLETED_RUNS = [
    {"name": "Faster R-CNN Baseline", "account": "amongus1504", "slug": "tod-aitod-baseline-s42-20260823", "cred": "kaggle (2).json", "tag": "baseline"},
    {"name": "NWD (NeurIPS 2021)", "account": "dipphmngc", "slug": "tod-aitod-nwd-s42-20260823", "cred": "kaggle (11).json", "tag": "nwd"},
    {"name": "RFLA (ECCV 2022)", "account": "luongsythanh", "slug": "tod-aitod-rfla-s42", "cred": "kaggle (8).json", "tag": "rfla"},
    {"name": "SAFit (AAAI 2024)", "account": "trieuvo123", "slug": "tod-aitod-safit-s42-20260824", "cred": "kaggle (10).json", "tag": "safit"},
    {"name": "H-WIoU (Proposed Ours)", "account": "amongus1504", "slug": "tod-h-wiou-sigma8-s42-20260820", "cred": "kaggle (2).json", "tag": "hwiou_sig8"},
]

def fetch_run(run: dict):
    name = run["name"]
    account = run["account"]
    slug = run["slug"]
    tag = run["tag"]
    cred_file = run["cred"]

    prof = PROFILE_ROOT / account
    prof.mkdir(parents=True, exist_ok=True)
    shutil.copy(CREDS_DIR / cred_file, prof / "kaggle.json")

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(prof)

    target_dir = DOWNLOAD_DIR / tag
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n--> Fetching output for {name} ({account}/{slug})...")
    cmd = [sys.executable, "-m", "kaggle", "kernels", "output", f"{account}/{slug}", "-p", str(target_dir)]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("Stdout:", res.stdout.strip())
    if res.stderr.strip():
        print("Stderr:", res.stderr.strip())

def main():
    print("=== DOWNLOADING COMPLETED OFFICIAL AI-TOD RUNS FROM KAGGLE ===")
    for run in COMPLETED_RUNS:
        fetch_run(run)

    print("\n=== SCANNING DOWNLOADED AI-TOD CHECKPOINTS AND LOGS ===")
    for f in DOWNLOAD_DIR.rglob("*"):
        if f.is_file() and (f.suffix in (".pt", ".pth", ".json", ".log", ".csv")):
            print(" ", f.relative_to(DOWNLOAD_DIR), f"({f.stat().st_size / 1e6:.2f} MB)")

if __name__ == "__main__":
    main()
