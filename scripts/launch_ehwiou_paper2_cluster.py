"""
Master Multi-Account Launcher for Paper 2: Entropy-Modulated Homotopy (EH-WIoU) & Cascade Cluster.
Dispatches parallel Tesla T4 GPU jobs across isolated Kaggle accounts.
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

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_ehwiou_profiles")
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR = ROOT / ".runtime/kaggle_ehwiou_cluster"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# Build hot-patch payload containing updated common, scripts, paper_a
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for root_dir in ["common", "scripts", "paper_a"]:
        src_dir = ROOT / root_dir
        for p in src_dir.rglob("*.py"):
            rel = p.relative_to(ROOT)
            z.write(p, str(rel).replace("\\", "/"))
b64_patch = base64.b64encode(buf.getvalue()).decode("utf-8")

JOBS = [
    # --- Paper 2 Core: Entropy-Modulated Homotopy (EH-WIoU) ---
    {
        "account": "amongus1504",
        "cred": CREDS_DIR / "kaggle (2).json",
        "dataset_type": "aitod",
        "tag": "aitod_ehwiou_sig8_s42",
        "slug": "tod-aitod-ehwiou-sig8-s42",
        "title": "TOD AI-TOD Entropy Homotopy EH-WIoU Sigma8 S42",
        "cmd": "python scripts/train_frcnn_aitod.py --metric eh_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 8.0 --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag ehwiou_sig8_s42",
    },
    {
        "account": "hienquang06",
        "cred": CREDS_DIR / "kaggle (5).json",
        "dataset_type": "aitod",
        "tag": "aitod_ehwiou_sig6_s42",
        "slug": "tod-aitod-ehwiou-sig6-s42",
        "title": "TOD AI-TOD Entropy Homotopy EH-WIoU Sigma6 S42",
        "cmd": "python scripts/train_frcnn_aitod.py --metric eh_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 6.0 --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag ehwiou_sig6_s42",
    },
    {
        "account": "dipphmngc",
        "cred": CREDS_DIR / "kaggle (11).json",
        "dataset_type": "tinyperson",
        "tag": "tp_ehwiou_sig8_s42",
        "slug": "tod-tp-ehwiou-sig8-s42",
        "title": "TOD TinyPerson Entropy Homotopy EH-WIoU Sigma8 S42",
        "cmd": "python scripts/train_frcnn_metric.py --metric eh_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 8.0 --seed 42 --tag ehwiou_sig8_s42",
    },
    # --- Cascade Homotopy Relaunches ---
    {
        "account": "phuc1806",
        "cred": CREDS_DIR / "kaggle (12).json",
        "dataset_type": "aitod",
        "tag": "aitod_cascade_homotopy_s42",
        "slug": "tod-cascade-homotopy-s42-proposed",
        "title": "TOD AI-TOD Cascade Multi-Stage Homotopy S42 Proposed",
        "cmd": "python scripts/train_cascade_aitod.py --metric h_wiou --placement la_loss --box-loss h_wiou --sigmas 8.0 4.0 2.0 --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag cascade_homotopy_s42",
    },
    {
        "account": "ngquangnht",
        "cred": CREDS_DIR / "kaggle (13).json",
        "dataset_type": "aitod",
        "tag": "aitod_rpn_cascade_hwiou_sig6_s42",
        "slug": "tod-aitod-rpn-cascade-hwiou-sig6-s42",
        "title": "TOD AI-TOD RPN Cascade H-WIoU Sigma6 S42 Proposed",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 6.0 --rpn-cascade --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag aitod_rpn_cascade_hwiou_sig6_s42",
    },
]


def launch_all():
    print("=" * 85)
    print("      DISPATCHING EH-WIOU & CASCADE BENCHMARK TO KAGGLE GPU CLUSTER      ")
    print("=" * 85)

    for job in JOBS:
        account = job["account"]
        tag = job["tag"]
        slug = job["slug"]
        title = job["title"]
        cmd_str = job["cmd"]
        dataset_type = job["dataset_type"]
        cred_file = job["cred"]

        # 1. Setup isolated Kaggle credential profile
        profile = PROFILE_ROOT / account
        profile.mkdir(parents=True, exist_ok=True)
        if not cred_file.exists():
            print(f"[SKIP] Credential file not found for {account}: {cred_file}")
            continue
        shutil.copy2(cred_file, profile / "kaggle.json")
        os.chmod(profile / "kaggle.json", 0o600)

        # 2. Setup Kaggle kernel working dir
        kdir = RUNTIME_DIR / tag
        kdir.mkdir(parents=True, exist_ok=True)

        if dataset_type == "tinyperson":
            dataset_sources = [
                "simplestzyp/tiny-object-detection-in-aerial-images",
                "quangnhatnguyen/tinypersondataset",
            ]
        else:
            dataset_sources = [
                "simplestzyp/tiny-object-detection-in-aerial-images",
            ]

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
            "dataset_sources": dataset_sources,
            "machine_shape": "NvidiaTeslaT4"
        }
        (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        # 3. Create notebook with hot-patch extractor
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
                    "import base64, io, os, sys, zipfile\n",
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
                    "print('Unpacked hot-patch successfully. Current dir:', os.getcwd())\n",
                    "print('Executing command:', " + json.dumps(cmd_str) + ")\n",
                    f"!{cmd_str}\n"
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

        # 4. Push to Kaggle via Kaggle CLI
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(profile)
        print(f"\n--> Pushing [{account}] {slug} ...")
        res = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kdir)],
            env=env, capture_output=True, text=True
        )
        print("Stdout:", res.stdout.strip())
        if res.stderr.strip():
            print("Stderr:", res.stderr.strip())

    print("\n" + "=" * 85)
    print("           ALL EH-WIOU & CASCADE EXPERIMENTS DISPATCHED SUCCESSFULLY          ")
    print("=" * 85)


if __name__ == "__main__":
    launch_all()
