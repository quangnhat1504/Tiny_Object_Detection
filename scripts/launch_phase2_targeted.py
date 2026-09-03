"""
Targeted Phase 2 Dispatcher:
1. Relaunch phuc1806 with memory-safe chunked EH-WIoU
2. Launch dipphmngc with verified tinyperson tiled dataset
3. Launch hngngnguynvn with multi-seed seed 123 for EH-WIoU
4. Launch trieuvo123 with Scale-Frequency Wavelet Homotopy (SW-HWIoU)
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
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_phase2_profiles")
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR = ROOT / ".runtime/kaggle_phase2_cluster"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# Package certified hot-patch payload with memory-safe chunking
print("[*] Packaging certified hot-patch zip payload...")
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for root_dir in ["common", "scripts", "paper_a"]:
        src_dir = ROOT / root_dir
        for p in src_dir.rglob("*.py"):
            rel = p.relative_to(ROOT)
            z.write(p, str(rel).replace("\\", "/"))
b64_patch = base64.b64encode(buf.getvalue()).decode("utf-8")
print(f"    Certified payload size: {len(b64_patch)/1024:.1f} KB")

TARGET_JOBS = [
    # 1. phuc1806: Memory-safe chunked EH-WIoU on AI-TOD-v2
    {
        "account": "phuc1806",
        "cred": CREDS_DIR / "kaggle (12).json",
        "dataset_sources": ["simplestzyp/tiny-object-detection-in-aerial-images"],
        "tag": "aitod_ehwiou_sig8_s42_phuc_v2",
        "slug": "tod-aitod-ehwiou-s42-proposed",
        "title": "TOD AI-TOD EH-WIoU Sigma8 S42 Proposed (Chunked Safe)",
        "cmd": "python scripts/train_frcnn_aitod.py --metric eh_wiou --placement h_wiou --box-loss eh_wiou --h-wiou-sigma-0 8.0 --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag ehwiou_sig8_s42",
    },
    # 2. dipphmngc: TinyPerson EH-WIoU with valid tiled dataset
    {
        "account": "dipphmngc",
        "cred": CREDS_DIR / "kaggle (11).json",
        "dataset_sources": ["dipphmngc/tod-program-b-tinyperson-b1-tiled-20260814"],
        "tag": "tp_ehwiou_sig8_s42_v2",
        "slug": "tod-tp-ehwiou-sig8-s42",
        "title": "TOD TinyPerson EH-WIoU Sigma8 S42 Proposed",
        "cmd": "python scripts/train_frcnn_metric.py --metric eh_wiou --placement h_wiou --box-loss eh_wiou --h-wiou-sigma-0 8.0 --seed 42 --tag ehwiou_sig8_s42",
    },
    # 3. hngngnguynvn: EH-WIoU Multi-Seed 123 on AI-TOD-v2
    {
        "account": "hngngnguynvn",
        "cred": CREDS_DIR / "kaggle (1).json",
        "dataset_sources": ["simplestzyp/tiny-object-detection-in-aerial-images"],
        "tag": "aitod_ehwiou_sig8_s123",
        "slug": "tod-aitod-ehwiou-sig8-s123",
        "title": "TOD AI-TOD EH-WIoU Sigma8 S123 Replication",
        "cmd": "python scripts/train_frcnn_aitod.py --metric eh_wiou --placement h_wiou --box-loss eh_wiou --h-wiou-sigma-0 8.0 --batch-size 4 --seed 123 --epochs 12 --eval-interval 3 --tag ehwiou_sig8_s123",
    },
    # 4. trieuvo123: Wavelet Homotopy SW-HWIoU on AI-TOD-v2
    {
        "account": "trieuvo123",
        "cred": CREDS_DIR / "kaggle (10).json",
        "dataset_sources": ["simplestzyp/tiny-object-detection-in-aerial-images"],
        "tag": "aitod_sw_hwiou_s42_proposed",
        "slug": "tod-aitod-sw-hwiou-s42-proposed",
        "title": "TOD AI-TOD Scale-Frequency Wavelet Homotopy SW-HWIoU S42 Proposed",
        "cmd": "python scripts/train_frcnn_aitod.py --metric sw_hwiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 8.0 --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag sw_hwiou_s42",
    },
]


def launch_phase2():
    print("=" * 85)
    print("           DISPATCHING PHASE 2 TARGETED JOBS WITH CERTIFIED HOT-PATCH          ")
    print("=" * 85)

    dispatched = []
    for job in TARGET_JOBS:
        account = job["account"]
        tag = job["tag"]
        slug = job["slug"]
        title = job["title"]
        cmd_str = job["cmd"]
        dataset_sources = job["dataset_sources"]
        cred_file = job["cred"]

        # Setup profile
        profile = PROFILE_ROOT / account
        profile.mkdir(parents=True, exist_ok=True)
        if not cred_file.exists():
            print(f"[SKIP] Credential missing for {account}: {cred_file}")
            continue
        shutil.copy2(cred_file, profile / "kaggle.json")
        os.chmod(profile / "kaggle.json", 0o600)

        # Working directory
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
            "dataset_sources": dataset_sources,
            "machine_shape": "NvidiaTeslaT4",
        }
        (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        # Notebook with strict error handling
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
                    "print('Unpacked certified hot-patch. Current dir:', os.getcwd())\n",
                    "cmd = " + json.dumps(cmd_str) + "\n",
                    "print('Executing command:', cmd)\n",
                    "res = subprocess.run(cmd, shell=True)\n",
                    "if res.returncode != 0:\n",
                    "    raise RuntimeError(f'Execution failed with return code {res.returncode}')\n",
                    "print('Execution completed successfully!')\n",
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
        print(f"\n--> Pushing [{account}] {slug} ...")
        res = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kdir)],
            env=env, capture_output=True, text=True
        )
        print("    Stdout:", res.stdout.strip())
        if res.stderr.strip():
            print("    Stderr:", res.stderr.strip())

        if res.returncode == 0:
            dispatched.append({"account": account, "slug": slug, "status": "PUSHED"})
        else:
            dispatched.append({"account": account, "slug": slug, "status": f"ERROR: {res.stderr[:30]}"})

    print("\n" + "=" * 85)
    print("             PHASE 2 TARGETED DISPATCH SUMMARY            ")
    print("=" * 85)
    for d in dispatched:
        print(f"  [{d['status']:<10}] {d['account']:<15} | {d['slug']}")
    print("=" * 85)


if __name__ == "__main__":
    launch_phase2()
