"""
Launch Full Empirical AI-TOD-v2 SOTA Benchmark Matrix on Kaggle Cluster (7 Active Accounts).
Methods:
1. Faster R-CNN Baseline (Standard IoU / Smooth-L1) [amongus1504]
2. NWD (NeurIPS 2021) [dipphmngc]
3. IGWD (IEEE TMM 2022) [hienquang06]
4. RFLA (ECCV 2022) [hngngnguynvn]
5. H-WIoU Proposed (sigma_0=8.0px) [quangnhtng]
6. H-WIoU Proposed (sigma_0=6.0px) [qnhat1504]
7. H-WIoU Proposed (sigma_0=10.0px) [thyngluthy]
"""
from __future__ import annotations
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_hwiou_profiles")
RUNTIME_DIR = ROOT / ".runtime/kaggle_aitod"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

AITOD_EXPERIMENTS = [
    {
        "tag": "aitod_baseline_s42",
        "title": "TOD AI-TOD Baseline Fair S42",
        "account": "amongus1504",
        "cred": "kaggle.json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric standard --placement iou_smooth_l1 --box-loss smooth_l1 --seed 42 --epochs 12 --batch-size 2",
    },
    {
        "tag": "aitod_nwd_s42",
        "title": "TOD AI-TOD NWD S42",
        "account": "dipphmngc",
        "cred": "kaggle (11).json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric nwd --placement everywhere --box-loss metric --seed 42 --epochs 12 --batch-size 2",
    },
    {
        "tag": "aitod_igwd_s42",
        "title": "TOD AI-TOD IGWD S42",
        "account": "hienquang06",
        "cred": "kaggle (5).json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric igwd --placement everywhere --box-loss metric --seed 42 --epochs 12 --batch-size 2",
    },
    {
        "tag": "aitod_rfla_s42",
        "title": "TOD AI-TOD RFLA S42",
        "account": "hngngnguynvn",
        "cred": "kaggle (1).json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric rfla --placement la --box-loss smooth_l1 --seed 42 --epochs 12 --batch-size 2",
    },
    {
        "tag": "aitod_hwiou_sig8_s42",
        "title": "TOD AI-TOD H-WIoU Sigma8 S42",
        "account": "quangnhtng",
        "cred": "kaggle (6).json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 8.0 --seed 42 --epochs 12 --batch-size 2",
    },
    {
        "tag": "aitod_hwiou_sig6_s42",
        "title": "TOD AI-TOD H-WIoU Sigma6 S42",
        "account": "qnhat1504",
        "cred": "kaggle (3).json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 6.0 --seed 42 --epochs 12 --batch-size 2",
    },
    {
        "tag": "aitod_hwiou_sig10_s42",
        "title": "TOD AI-TOD H-WIoU Sigma10 S42",
        "account": "thyngluthy",
        "cred": "kaggle (4).json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 10.0 --seed 42 --epochs 12 --batch-size 2",
    },
]


def build_patch_payload() -> str:
    """Build lightweight hot-patch zip payload containing all updated files."""
    files = [
        "common/model.py",
        "common/metrics/iou.py",
        "common/metrics/__init__.py",
        "scripts/train_frcnn_aitod.py",
        "paper_a/evaluation/aitodv2_official.py",
        "paper_a/datasets/aitodv2_adapter.py",
        "paper_a/datasets/coco_original.py",
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            p = ROOT / f
            if p.exists():
                z.write(p, f)
        for p in (ROOT / "common/metrics").glob("*.py"):
            z.write(p, f"common/metrics/{p.name}")
    b64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
    print(f"  -> Hot-patch payload: {len(b64_str)} chars ({len(buf.getvalue()) / 1024:.1f} KB)")
    return b64_str


def launch_aitod_experiment(exp: dict, b64_patch: str):
    account = exp["account"]
    tag = exp["tag"]
    title = exp["title"]
    cmd_str = exp["cmd"]
    slug = f"tod-{tag.replace('_', '-')}-20260823"

    kernel_dir = RUNTIME_DIR / tag
    kernel_dir.mkdir(parents=True, exist_ok=True)

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
            f"{account}/tod-program-b-b2-code-20260814",
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    f"# {title}\n",
                    f"Automated AI-TOD-v2 empirical training: `{cmd_str}`\n"
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os, sys, shutil, subprocess, json, torch, base64, io, zipfile\n",
                    "from pathlib import Path\n",
                    "\n",
                    f"tag = '{tag}'\n",
                    f"account = '{account}'\n",
                    "\n",
                    "# 1. Setup workspace from mounted code dataset\n",
                    f"code_src = Path(f'/kaggle/input/datasets/{account}/tod-program-b-b2-code-20260814')\n",
                    "if not code_src.exists():\n",
                    "    candidates = list(Path('/kaggle/input').glob(f'*{account}*')) + list(Path('/kaggle/input').glob('*tod-program*'))\n",
                    "    if candidates:\n",
                    "        code_src = candidates[0]\n",
                    "\n",
                    f"work_dir = Path(f'/kaggle/working/{tag}')\n",
                    "if work_dir.exists():\n",
                    "    shutil.rmtree(work_dir)\n",
                    "shutil.copytree(code_src, work_dir)\n",
                    "os.chdir(work_dir)\n",
                    "sys.path.insert(0, str(work_dir))\n",
                    "\n",
                    "# 2. Apply latest hot-patch fixes\n",
                    f"patch_b64 = '''{b64_patch}'''\n",
                    "patch_zip = base64.b64decode(patch_b64)\n",
                    "with zipfile.ZipFile(io.BytesIO(patch_zip)) as z:\n",
                    "    z.extractall(work_dir)\n",
                    "print(f'Applied latest hot-patch fixes to {work_dir}')\n",
                    "\n",
                    "# 3. Setup PyTorch cache if available\n",
                    "torch_cache = work_dir / 'torch_cache'\n",
                    "if torch_cache.exists():\n",
                    "    dst_cache = Path.home() / '.cache/torch'\n",
                    "    dst_cache.mkdir(parents=True, exist_ok=True)\n",
                    "    for f in (torch_cache / 'hub/checkpoints').glob('*.pth'):\n",
                    "        (dst_cache / 'hub/checkpoints').mkdir(parents=True, exist_ok=True)\n",
                    "        shutil.copy(f, dst_cache / 'hub/checkpoints' / f.name)\n",
                    "\n",
                    "# 4. Install dependencies\n",
                    "subprocess.run(['pip', 'install', '-q', 'pycocotools', 'torchmetrics'], check=False)\n",
                    "\n",
                    "# 5. Execute official training\n",
                    f"print('Executing command: {cmd_str}')\n",
                    f"proc = subprocess.run({repr(cmd_str.split())}, stdout=sys.stdout, stderr=sys.stderr)\n",
                    "print('Execution finished with exit code:', proc.returncode)\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    (kernel_dir / f"{tag}.ipynb").write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")

    # Set Kaggle Auth Profile
    profile = PROFILE_ROOT / account
    profile.mkdir(parents=True, exist_ok=True)
    shutil.copy(CREDS_DIR / exp["cred"], profile / "kaggle.json")
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(profile)

    print(f"\n[{account}] Pushing notebook kernel {meta['id']}...")
    push_cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kernel_dir)]
    res = subprocess.run(push_cmd, env=env, capture_output=True, text=True)
    print("Push Output:", res.stdout.strip())
    if res.stderr.strip():
        print("Push Stderr:", res.stderr.strip())

    time.sleep(2)
    stat_cmd = [sys.executable, "-m", "kaggle", "kernels", "status", meta["id"]]
    stat_res = subprocess.run(stat_cmd, env=env, capture_output=True, text=True)
    print(f"Status: {stat_res.stdout.strip()}")


def main():
    print("=== Launching Full Empirical AI-TOD-v2 Benchmark Matrix on Kaggle ===")
    b64_patch = build_patch_payload()
    for exp in AITOD_EXPERIMENTS:
        launch_aitod_experiment(exp, b64_patch)
    print("\n=== All 7 AI-TOD-v2 Experiments Dispatched Successfully! ===")


if __name__ == "__main__":
    main()
