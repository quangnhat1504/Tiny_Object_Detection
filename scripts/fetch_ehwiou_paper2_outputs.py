"""
Download outputs and logs from completed Paper 2 EH-WIoU & Cascade Kaggle jobs.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_ehwiou_profiles")
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = ROOT / "runs/ehwiou_paper2_kaggle_outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_JOBS = [
    {
        "account": "amongus1504",
        "cred": CREDS_DIR / "kaggle (2).json",
        "slug": "tod-aitod-ehwiou-sig8-s42",
        "tag": "aitod_ehwiou_sig8_s42",
    },
    {
        "account": "hienquang06",
        "cred": CREDS_DIR / "kaggle (5).json",
        "slug": "tod-aitod-ehwiou-sig6-s42",
        "tag": "aitod_ehwiou_sig6_s42",
    },
    {
        "account": "dipphmngc",
        "cred": CREDS_DIR / "kaggle (11).json",
        "slug": "tod-tp-ehwiou-sig8-s42",
        "tag": "tp_ehwiou_sig8_s42",
    },
    {
        "account": "phuc1806",
        "cred": CREDS_DIR / "kaggle (12).json",
        "slug": "tod-cascade-homotopy-s42-proposed",
        "tag": "aitod_cascade_homotopy_s42",
    },
]


def fetch_all():
    print("=" * 80)
    print("      DOWNLOADING ARTIFACTS FROM COMPLETED KAGGLE KERNELS      ")
    print("=" * 80)

    for job in TARGET_JOBS:
        account = job["account"]
        slug = job["slug"]
        cred_file = job["cred"]
        tag = job["tag"]

        profile = PROFILE_ROOT / account
        profile.mkdir(parents=True, exist_ok=True)
        target_cred = profile / "kaggle.json"
        if not target_cred.exists() and cred_file.exists():
            shutil.copy(cred_file, target_cred)

        dest = OUTPUT_DIR / tag
        dest.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(profile)

        print(f"\n[*] Fetching outputs for {account}/{slug} -> {dest}")
        res = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "output", f"{account}/{slug}", "-p", str(dest)],
            env=env, capture_output=True, text=True
        )
        print(f"    Stdout: {res.stdout.strip()}")
        if res.stderr.strip():
            print(f"    Stderr: {res.stderr.strip()}")

    print("\n" + "=" * 80)
    print("Download completed! Inspecting downloaded artifacts:")
    for job in TARGET_JOBS:
        tag = job["tag"]
        dest = OUTPUT_DIR / tag
        files = list(dest.glob("*"))
        print(f"  - {tag} ({len(files)} files): {[f.name for f in files]}")
    print("=" * 80)


if __name__ == "__main__":
    fetch_all()
