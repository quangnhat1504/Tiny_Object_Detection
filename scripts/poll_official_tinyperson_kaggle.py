"""
Poll status of 7 official TinyPerson runs on Kaggle across 7 dedicated accounts with matching datasets.
"""
import json
import os
import subprocess
from pathlib import Path

PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_official_tp_profiles")

KERNELS = [
    ("Faster R-CNN Baseline", "qnhat1504", "tp-official-standard-s42-fair20"),
    ("NWD (NeurIPS 2021)", "luongsythanh", "tp-official-tp-official-nwd-s42-fair2"),
    ("IGWD (IEEE TMM 2022)", "thyngluthy", "tp-official-tp-official-igwd-s42-fair2"),
    ("SA-ALW (Paper A)", "pptlyn11", "tp-official-tp-official-saalw-s42-fair2"),
    ("H-WIoU (sig=8.0)", "hngtrngtn", "tp-official-tp-official-hwiou-sig8-s42-fair2"),
    ("H-WIoU (sig=6.0)", "hngngnguynvn", "tp-official-tp-official-hwiou-sig6-s42-fair2"),
    ("RFLA (ECCV 2022)", "dipphmngc", "tp-official-tp-official-rfla-s42-fair2"),
]

def main():
    print("=" * 80)
    print("=== STATUS OF 7 OFFICIAL TINYPERSON RUNS ON KAGGLE GPU POOL ===")
    print("=" * 80)

    for name, account, kernel_slug in KERNELS:
        profile = PROFILE_ROOT / account
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(profile)
        
        full_slug = f"{account}/{kernel_slug}"
        cmd = ["kaggle", "kernels", "status", full_slug]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        status = res.stdout.strip() if res.returncode == 0 else res.stderr.strip()
        print(f"[{account:<14}] {name:<25}: {status}")

if __name__ == "__main__":
    main()
