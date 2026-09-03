"""
Master Poller for Entire 8-Worker Active Kaggle GPU Matrix.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
HEALED_PROFILES = Path(r"C:\tmp\tod_kaggle_healed_profiles")
PHASE2_PROFILES = Path(r"C:\tmp\tod_kaggle_phase2_profiles")

ACTIVE_JOBS = [
    # Group 1: Healed Cluster
    {"account": "qnhat1504", "slug": "tod-aitod-rpn-cascade-hwiou-sig6-s42", "prof": HEALED_PROFILES / "qnhat1504", "desc": "RPN Cascade H-WIoU S42"},
    {"account": "hienquang06", "slug": "tod-cascade-homotopy-s42-proposed", "prof": HEALED_PROFILES / "hienquang06", "desc": "Cascade Homotopy Multi-Stage S42"},
    {"account": "thyngluthy", "slug": "tod-aitod-ehwiou-sig6-s42", "prof": HEALED_PROFILES / "thyngluthy", "desc": "AI-TOD EH-WIoU Sigma6 S42"},
    {"account": "amongus1504", "slug": "tod-aitod-qfl-duhwiou-s42-proposed", "prof": HEALED_PROFILES / "amongus1504", "desc": "QFL + DU-HWIoU Proposed S42"},
    # Group 2: Phase 2 Targeted Cluster
    {"account": "phuc1806", "slug": "tod-aitod-ehwiou-s42-proposed", "prof": PHASE2_PROFILES / "phuc1806", "desc": "AI-TOD EH-WIoU Sigma8 S42 Proposed (Chunked Safe)"},
    {"account": "dipphmngc", "slug": "tod-tp-ehwiou-sig8-s42", "prof": PHASE2_PROFILES / "dipphmngc", "desc": "TinyPerson EH-WIoU Sigma8 S42"},
    {"account": "hngngnguynvn", "slug": "tod-aitod-ehwiou-sig8-s123", "prof": PHASE2_PROFILES / "hngngnguynvn", "desc": "AI-TOD EH-WIoU Sigma8 S123 Replication"},
    {"account": "trieuvo123", "slug": "tod-aitod-sw-hwiou-s42-proposed", "prof": PHASE2_PROFILES / "trieuvo123", "desc": "Wavelet Homotopy SW-HWIoU S42 Proposed"},
]


def poll_matrix():
    print("=" * 105)
    print("           REAL-TIME STATUS REPORT: 8 CONCURRENT KAGGLE GPU ACCELERATOR WORKERS           ")
    print("=" * 105)

    summary = []
    for item in ACTIVE_JOBS:
        account = item["account"]
        slug = item["slug"]
        prof = item["prof"]
        desc = item["desc"]

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

        print(f"  [{status:<10}] {account:<15} | {slug:<38} | {desc}")
        summary.append({"account": account, "slug": slug, "status": status, "desc": desc})

    print("=" * 105)
    return summary


if __name__ == "__main__":
    poll_matrix()
