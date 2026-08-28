"""
Download all 11 completed AI-TOD-v2 model checkpoints from the Kaggle cluster.
"""
from __future__ import annotations
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CREDS_DIR = Path.home() / ".kaggle"
DOWNLOAD_DIR = ROOT / "runs" / "aitod_kaggle_checkpoints"
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

from paper_a.tools.monitor_and_download_aitod_matrix import ACTIVE_CLUSTER_KERNELS, download_output

def main():
    print("=" * 80)
    print("       DOWNLOADING ALL 11 COMPLETED AI-TOD-V2 CHECKPOINTS FROM KAGGLE       ")
    print("=" * 80)
    for idx, item in enumerate(ACTIVE_CLUSTER_KERNELS, 1):
        account = item["account"]
        slug = item["slug"]
        tag = item["tag"]
        print(f"[{idx:02d}/11] Downloading {account}/{slug} ({tag}) ...", flush=True)
        download_output(item)
    print("=" * 80)
    print("       ALL 11 CHECKPOINTS DOWNLOADED SUCCESSFULLY TO runs/aitod_kaggle_checkpoints/       ")
    print("=" * 80)

if __name__ == "__main__":
    main()
