"""
Dispatch Certified EH-WIoU Proposed Experiment to Kaggle GPU Pool (Account: phuc1806).
Uses certified hot-patch payload with subprocess error checking.
"""
from __future__ import annotations
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_ehwiou_profiles")
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR = ROOT / ".runtime/kaggle_ehwiou_cluster"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# 1. Build certified hot-patch payload
print("[*] Packaging certified hot-patch zip payload...")
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for root_dir in ["common", "scripts", "paper_a"]:
        src_dir = ROOT / root_dir
        for p in src_dir.rglob("*.py"):
            rel = p.relative_to(ROOT)
            z.write(p, str(rel).replace("\\", "/"))
b64_patch = base64.b64encode(buf.getvalue()).decode("utf-8")
print(f"    Payload size: {len(b64_patch)/1024:.1f} KB")

# Target Job Configuration for phuc1806
JOB = {
    "account": "phuc1806",
    "cred": CREDS_DIR / "kaggle (12).json",
    "tag": "aitod_ehwiou_sig8_s42_phuc",
    "slug": "tod-aitod-ehwiou-s42-proposed",
    "title": "TOD AI-TOD Entropy Homotopy EH-WIoU Sigma8 S42 Proposed",
    "cmd": "python scripts/train_frcnn_aitod.py --metric eh_wiou --placement h_wiou --box-loss eh_wiou --h-wiou-sigma-0 8.0 --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag ehwiou_sig8_s42",
}


def dispatch_kernel():
    account = JOB["account"]
    tag = JOB["tag"]
    slug = JOB["slug"]
    title = JOB["title"]
    cmd_str = JOB["cmd"]
    cred_file = JOB["cred"]

    print("=" * 85)
    print(f"   DISPATCHING EXPERIMENT TO KAGGLE: {account}/{slug}")
    print("=" * 85)

    # Setup profile
    profile = PROFILE_ROOT / account
    profile.mkdir(parents=True, exist_ok=True)
    if not cred_file.exists():
        raise FileNotFoundError(f"Credential file not found: {cred_file}")
    shutil.copy2(cred_file, profile / "kaggle.json")
    os.chmod(profile / "kaggle.json", 0o600)

    # Kernel directory
    kdir = RUNTIME_DIR / tag
    kdir.mkdir(parents=True, exist_ok=True)

    meta = {
        "id": f"{account}/{slug}",
        "title": slug,
        "code_file": f"{tag}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
        "machine_shape": "NvidiaTeslaT4"
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    # Build notebook with strict subprocess check
    nb_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"# {title}\n", f"Command: `{cmd_str}`\n"]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import base64, io, os, sys, zipfile, subprocess\n",
                "from pathlib import Path\n",
                "\n",
                f"PATCH_B64 = '{b64_patch}'\n",
                "buf = io.BytesIO(base64.b64decode(PATCH_B64))\n",
                "with zipfile.ZipFile(buf, 'r') as z:\n",
                "    z.extractall('/kaggle/working')\n",
                "\n",
                "!pip install -q pycocotools terminaltables\n",
                "\n",
                "os.chdir('/kaggle/working')\n",
                "print('Unpacked certified hot-patch successfully. Current dir:', os.getcwd())\n",
                "cmd = " + json.dumps(cmd_str) + "\n",
                "print('Executing command:', cmd)\n",
                "res = subprocess.run(cmd, shell=True)\n",
                "if res.returncode != 0:\n",
                "    raise RuntimeError(f'Execution failed with code {res.returncode}')\n",
                "print('Training finished successfully!')\n",
            ]
        }
    ]

    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    (kdir / f"{tag}.ipynb").write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")

    # Push kernel
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(profile)
    print(f"\n--> Pushing [{account}] {slug} to Kaggle...")
    res = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kdir)],
        env=env, capture_output=True, text=True
    )
    print("Stdout:", res.stdout.strip())
    if res.stderr.strip():
        print("Stderr:", res.stderr.strip())

    if res.returncode != 0:
        raise RuntimeError(f"Kaggle kernel push failed: {res.stderr or res.stdout}")

    print(f"[SUCCESS] Kernel {account}/{slug} pushed successfully!")
    print("=" * 85)


if __name__ == "__main__":
    dispatch_kernel()
