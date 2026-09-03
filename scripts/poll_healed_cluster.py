"""
Poll Live Status of All Active Kaggle GPU Jobs across Healed Cluster.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_healed_profiles")
PHUC_PROFILE = Path(r"C:\tmp\tod_kaggle_ehwiou_profiles\phuc1806")

from scripts.launch_healed_cluster import JOBS

ALL_JOBS = list(JOBS) + [
    {
        "account": "phuc1806",
        "slug": "tod-aitod-ehwiou-s42-proposed",
        "title": "EH-WIoU Proposed S42 (Pilot)",
    }
]


def poll_all():
    print("=" * 90)
    print("           POLLING LIVE ACCELERATOR STATUS ACROSS HEALED KAGGLE CLUSTER           ")
    print("=" * 90)

    summary = []
    for job in ALL_JOBS:
        account = job["account"]
        slug = job["slug"]

        if account == "phuc1806":
            prof = PHUC_PROFILE
        else:
            prof = PROFILE_ROOT / account

        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(prof)

        res = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "status", f"{account}/{slug}"],
            env=env, capture_output=True, text=True, timeout=20
        )
        out = (res.stdout + " " + res.stderr).strip()
        status = "UNKNOWN"
        for s in ["running", "complete", "error", "queued", "cancel_acknowledged"]:
            if s in out.lower():
                status = s.upper()
                break

        print(f"  [{status:<10}] {account:<15} | {slug:<38} | {out[:30]}")
        summary.append({"account": account, "slug": slug, "status": status})

    print("=" * 90)
    return summary


if __name__ == "__main__":
    poll_all()
