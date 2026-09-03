"""
Poll Live Status and Retrieve Metrics/Checkpoints for Direction 1 Generalization Cluster across 13 Kaggle Accounts.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CREDS_DIR = Path.home() / ".kaggle"
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_gen_profiles")
RUNTIME_DIR = ROOT / ".runtime/kaggle_generalization_cluster"

from scripts.launch_direction1_generalization_cluster import JOBS

def check_status():
    print("=" * 90)
    print("         POLLING 13 KAGGLE WORKERS FOR DIRECTION 1 GENERALIZATION EXPERIMENTS         ")
    print("=" * 90)

    results = []

    for job in JOBS:
        account = job["account"]
        tag = job["tag"]
        slug = job["slug"]
        cred_file = job["cred"]

        profile = PROFILE_ROOT / account
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(profile)

        res = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "status", f"{account}/{slug}"],
            env=env, capture_output=True, text=True
        )
        out = (res.stdout + " " + res.stderr).strip()
        status = "UNKNOWN"
        for st in ["running", "complete", "error", "queued"]:
            if f'"{st}"' in out.lower() or f"'{st}'" in out.lower() or st in out.lower():
                status = st.upper()
                break

        print(f"  [{status:<8}] {account:<15} | {slug:<38} | {out[:30]}")
        results.append({"account": account, "tag": tag, "slug": slug, "status": status})

    print("=" * 90)
    return results

if __name__ == "__main__":
    check_status()
