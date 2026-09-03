"""
Polling Script for Paper 2 EH-WIoU & Cascade Kaggle GPU Jobs.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_ehwiou_profiles")

from scripts.launch_ehwiou_paper2_cluster import JOBS


def poll_status():
    print("=" * 85)
    print("      POLLING LIVE STATUS FOR PAPER 2 EH-WIOU & CASCADE KAGGLE GPU JOBS      ")
    print("=" * 85)

    all_done = True
    for job in JOBS:
        account = job["account"]
        slug = job["slug"]
        tag = job["tag"]
        profile = PROFILE_ROOT / account

        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(profile)

        res = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "status", f"{account}/{slug}"],
            env=env, capture_output=True, text=True
        )
        status_raw = (res.stdout + " " + res.stderr).strip()
        status = "UNKNOWN"
        for st in ["running", "complete", "error", "queued", "cancel_acknowledged"]:
            if f'"{st}"' in status_raw.lower() or f"'{st}'" in status_raw.lower() or st in status_raw.lower():
                status = st.upper()
                break

        print(f"  [{status:<8}] {account:<14} | {slug:<38} | {status_raw[:35]}")
        if status != "COMPLETE":
            all_done = False

    print("=" * 85)
    return all_done


if __name__ == "__main__":
    poll_status()
