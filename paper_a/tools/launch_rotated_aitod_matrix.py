"""
Push remaining rotated AI-TOD-v2 tasks to active accounts with available GPU quotas.
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

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_rotation_profiles")
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR = ROOT / ".runtime/kaggle_aitod_rotation"
RUNTIME_DIR.mkdir(parents=True, exist_ok=True)

# Build hot-patch payload
buf = io.BytesIO()
with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
    for root_dir in ["common", "scripts", "paper_a"]:
        src_dir = ROOT / root_dir
        for p in src_dir.rglob("*.py"):
            rel = p.relative_to(ROOT)
            z.write(p, str(rel).replace("\\", "/"))
b64_patch = base64.b64encode(buf.getvalue()).decode("utf-8")

ROTATION_EXPS = [
    {
        "tag": "aitod_hwiou_sig8_s42_rot",
        "slug": "tod-aitod-hwiou-sig8-s42-rot",
        "title": "TOD AI-TOD H-WIoU Sigma8 S42 Proposed",
        "account": "amongus1504",
        "cred": CREDS_DIR / "kaggle.json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 8.0 --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag hwiou_sig8_s42",
    },
    {
        "tag": "aitod_hwiou_sig6_s42_rot",
        "slug": "tod-aitod-hwiou-sig6-s42-rot",
        "title": "TOD AI-TOD H-WIoU Sigma6 S42 Ablation",
        "account": "dipphmngc",
        "cred": CREDS_DIR / "kaggle (11).json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 6.0 --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag hwiou_sig6_s42",
    },
    {
        "tag": "aitod_igwd_s42_rot",
        "slug": "tod-aitod-igwd-s42-rot",
        "title": "TOD AI-TOD IGWD S42",
        "account": "thyngluthy",
        "cred": CREDS_DIR / "kaggle (4).json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric igwd --placement everywhere --box-loss metric --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag igwd_s42",
    },
    {
        "tag": "aitod_rfla_s42_rot",
        "slug": "tod-aitod-rfla-s42-rot",
        "title": "TOD AI-TOD RFLA S42",
        "account": "phuc1806",
        "cred": CREDS_DIR / "kaggle (12).json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric rfla --placement la --box-loss smooth_l1 --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag rfla_s42",
    },
]

def main():
    print("=" * 80)
    print("      LAUNCHING ROTATED SOTA BENCHMARKS ON ACTIVE GPU ACCOUNTS      ")
    print("=" * 80)

    for exp in ROTATION_EXPS:
        account = exp["account"]
        tag = exp["tag"]
        slug = exp["slug"]
        title = exp["title"]
        cmd_str = exp["cmd"]
        cred_file = exp["cred"]

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
            "dataset_sources": ["simplestzyp/tiny-object-detection-in-aerial-images"],
            "kernel_sources": [],
            "competition_sources": [],
            "model_sources": [],
            "machine_shape": "NvidiaTeslaT4",
        }
        (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

        nb = {
            "cells": [
                {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n", f"Command: `{cmd_str}`\n"]},
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import os, sys, shutil, subprocess, json, torch, base64, io, zipfile\n",
                        "from pathlib import Path\n",
                        f"tag = '{tag}'\n",
                        f"account = '{account}'\n",
                        "print(f'=== Starting {tag} on {account} (CUDA: {torch.cuda.is_available()}) ===')\n",
                        f"work_dir = Path(f'/kaggle/working/{tag}')\n",
                        "if work_dir.exists(): shutil.rmtree(work_dir)\n",
                        "work_dir.mkdir(parents=True, exist_ok=True)\n",
                        "os.chdir(work_dir)\n",
                        "sys.path.insert(0, str(work_dir))\n",
                        f"patch_b64 = '''{b64_patch}'''\n",
                        "with zipfile.ZipFile(io.BytesIO(base64.b64decode(patch_b64))) as z: z.extractall(work_dir)\n",
                        "subprocess.run(['pip', 'install', '-q', 'pycocotools', 'torchmetrics'], check=False)\n",
                        f"print('Executing: {cmd_str}')\n",
                        f"proc = subprocess.run({repr(cmd_str.split())}, stdout=sys.stdout, stderr=sys.stderr)\n",
                        "print('Exit code:', proc.returncode)\n",
                        "runs_dir = work_dir / 'runs'\n",
                        "if runs_dir.exists():\n",
                        "    out_target = Path('/kaggle/working') / 'tod_output'\n",
                        "    out_target.mkdir(parents=True, exist_ok=True)\n",
                        "    shutil.copytree(runs_dir, out_target / 'runs', dirs_exist_ok=True)\n",
                    ]
                }
            ],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.10.12"}},
            "nbformat": 4, "nbformat_minor": 2,
        }
        (kdir / f"{tag}.ipynb").write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")

        prof = PROFILE_ROOT / account
        prof.mkdir(parents=True, exist_ok=True)
        shutil.copy(cred_file, prof / "kaggle.json")
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(prof)

        print(f"Pushing {account}/{slug} ...", end=" ", flush=True)
        res = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kdir)], env=env, capture_output=True, text=True)
        print(res.stdout.strip() if res.stdout.strip() else res.stderr.strip())

if __name__ == "__main__":
    main()
