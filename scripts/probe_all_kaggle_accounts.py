"""
Probe all 13 Kaggle accounts to test GPU launch availability.
"""
from __future__ import annotations
import os
import sys
import shutil
import subprocess
import json
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"
PROBE_DIR = ROOT / ".runtime/kaggle_probe"
PROBE_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNTS = [
    ("amongus1504", CREDS_DIR / "kaggle.json"),
    ("dipphmngc", CREDS_DIR / "kaggle (11).json"),
    ("hienquang06", CREDS_DIR / "kaggle (5).json"),
    ("hngngnguynvn", CREDS_DIR / "kaggle (1).json"),
    ("hngtrngtn", CREDS_DIR / "kaggle (7).json"),
    ("luongsythanh", CREDS_DIR / "kaggle (8).json"),
    ("ngquangnht", ROOT / ".runtime/kaggle/wp02/multi_account/cfg_ngquangnht/kaggle.json"),
    ("phuc1806", CREDS_DIR / "kaggle (12).json"),
    ("pptlyn11", CREDS_DIR / "kaggle (9).json"),
    ("qnhat1504", CREDS_DIR / "kaggle (3).json"),
    ("quangnhtng", CREDS_DIR / "kaggle (6).json"),
    ("thyngluthy", CREDS_DIR / "kaggle (4).json"),
    ("trieuvo123", CREDS_DIR / "kaggle (10).json"),
]

def main():
    print("=" * 80)
    print("      PROBING GPU SLOT AVAILABILITY ACROSS ALL 13 KAGGLE ACCOUNTS      ")
    print("=" * 80)

    available = []
    busy_or_limited = []

    for account, cred in ACCOUNTS:
        if not cred.exists():
            continue
        profile = Path(r"C:\tmp\tod_probe") / account
        profile.mkdir(parents=True, exist_ok=True)
        shutil.copy(cred, profile / "kaggle.json")
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(profile)

        slug = f"test-probe-{account}"
        kdir = PROBE_DIR / account
        kdir.mkdir(parents=True, exist_ok=True)
        meta = {
            "id": f"{account}/{slug}",
            "title": slug,
            "code_file": f"{slug}.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
            "dataset_sources": [],
            "machine_shape": "NvidiaTeslaT4"
        }
        (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        nb = {
            "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": ["print('GPU Probe OK')\n"]}],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
            "nbformat": 4, "nbformat_minor": 2
        }
        (kdir / f"{slug}.ipynb").write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")

        res = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kdir)], env=env, capture_output=True, text=True)
        out = (res.stdout + " " + res.stderr).strip()

        if "successfully pushed" in out:
            print(f"  [AVAILABLE] {account:<15}: SUCCESS (GPU Slot Open!)")
            available.append(account)
        else:
            reason = "Quota limit (reset in ~24h)" if "quota" in out.lower() else out[:60]
            print(f"  [BUSY/LIMIT] {account:<15}: {reason}")
            busy_or_limited.append((account, reason))

    print("\n" + "=" * 80)
    print(f"Summary: {len(available)} Accounts Available | {len(busy_or_limited)} Accounts Busy/Resting")
    print("=" * 80)

if __name__ == "__main__":
    main()
