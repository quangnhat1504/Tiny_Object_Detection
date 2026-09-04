"""
Master Fetcher for Completed Kaggle Matrix Jobs and Error Diagnostics.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
HEALED_PROFILES = Path(r"C:\tmp\tod_kaggle_healed_profiles")
PHASE2_PROFILES = Path(r"C:\tmp\tod_kaggle_phase2_profiles")
OUTPUT_BASE = ROOT / "runs/matrix_kaggle_outputs"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

FETCH_TARGETS = [
    {"account": "thyngluthy", "slug": "tod-aitod-ehwiou-sig6-s42", "prof": HEALED_PROFILES / "thyngluthy", "tag": "aitod_ehwiou_sig6_s42"},
    {"account": "amongus1504", "slug": "tod-aitod-qfl-duhwiou-s42-proposed", "prof": HEALED_PROFILES / "amongus1504", "tag": "aitod_qfl_duhwiou_s42"},
    {"account": "phuc1806", "slug": "tod-aitod-ehwiou-s42-proposed", "prof": PHASE2_PROFILES / "phuc1806", "tag": "aitod_ehwiou_s42_chunked"},
    {"account": "hngngnguynvn", "slug": "tod-aitod-ehwiou-sig8-s123", "prof": PHASE2_PROFILES / "hngngnguynvn", "tag": "aitod_ehwiou_sig8_s123"},
    {"account": "trieuvo123", "slug": "tod-aitod-sw-hwiou-s42-proposed", "prof": PHASE2_PROFILES / "trieuvo123", "tag": "aitod_sw_hwiou_s42"},
    {"account": "dipphmngc", "slug": "tod-tp-ehwiou-sig8-s42", "prof": PHASE2_PROFILES / "dipphmngc", "tag": "tp_ehwiou_sig8_s42_diag"},
]


def fetch_all():
    print("=" * 80)
    print("      DOWNLOADING ARTIFACTS AND LOGS FROM COMPLETED KAGGLE JOBS      ")
    print("=" * 80)

    for item in FETCH_TARGETS:
        account = item["account"]
        slug = item["slug"]
        prof = item["prof"]
        tag = item["tag"]

        dest = OUTPUT_BASE / tag
        dest.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(prof)

        print(f"\n[*] Downloading {account}/{slug} -> {dest}")
        res = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "output", f"{account}/{slug}", "-p", str(dest)],
            env=env, capture_output=True, text=True
        )
        if res.stdout.strip():
            print(f"    Stdout: {res.stdout.strip()}")
        if res.stderr.strip():
            print(f"    Stderr: {res.stderr.strip()}")

    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY:")
    for item in FETCH_TARGETS:
        tag = item["tag"]
        dest = OUTPUT_BASE / tag
        files = list(dest.rglob("*"))
        file_names = [f.relative_to(dest).as_posix() for f in files if f.is_file()]
        print(f"  [{tag}] {len(file_names)} files: {file_names[:8]}")
    print("=" * 80)


if __name__ == "__main__":
    fetch_all()
