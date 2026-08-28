"""
Push all remaining Fair Unified AI-TOD-v2 benchmark jobs to the highest-quota Kaggle accounts.
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
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_fair_profiles")
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR = ROOT / ".runtime/kaggle_aitod_fair"
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

FAIR_JOBS = [
    {
        "tag": "aitod_hwiou_sig8_fair42",
        "slug": "tod-aitod-hwiou-sig8-fair42",
        "title": "TOD AI-TOD H-WIoU Sigma8 Fair Proposed",
        "account": "hngngnguynvn",
        "cred": CREDS_DIR / "kaggle (1).json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 8.0 --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag hwiou_sig8_fair42",
    },
    {
        "tag": "aitod_hwiou_sig6_fair42",
        "slug": "tod-aitod-hwiou-sig6-fair42",
        "title": "TOD AI-TOD H-WIoU Sigma6 Fair Ablation",
        "account": "ngquangnht",
        "cred": ROOT / ".runtime/kaggle/wp02/multi_account/cfg_ngquangnht/kaggle.json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 6.0 --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag hwiou_sig6_fair42",
    },
    {
        "tag": "aitod_hwiou_sig10_fair42",
        "slug": "tod-aitod-hwiou-sig10-fair42",
        "title": "TOD AI-TOD H-WIoU Sigma10 Fair Ablation",
        "account": "pptlyn11",
        "cred": CREDS_DIR / "kaggle (9).json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 10.0 --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag hwiou_sig10_fair42",
    },
    {
        "tag": "aitod_nwd_fair42",
        "slug": "tod-aitod-nwd-fair42",
        "title": "TOD AI-TOD NWD Fair Assigner Loss",
        "account": "hngtrngtn",
        "cred": CREDS_DIR / "kaggle (7).json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric nwd --placement la_loss --box-loss metric --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag nwd_fair42",
    },
]

def main():
    print("=" * 80)
    print("      LAUNCHING FAIR BENCHMARKS ON HIGH-QUOTA KAGGLE GPU ACCOUNTS      ")
    print("=" * 80)

    for exp in FAIR_JOBS:
        account = exp["account"]
        tag = exp["tag"]
        slug = exp["slug"]
        title = exp["title"]
        cmd_str = exp["cmd"]
        cred_file = exp["cred"]

        if not cred_file.exists():
            print(f"Skipping {account}: Cred file {cred_file} not found")
            continue

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
            "machine_shape": "NvidiaTeslaT4"
        }
        (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

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
                    "import os, sys, shutil, subprocess, json, base64, io, zipfile\n",
                    "from pathlib import Path\n",
                    "\n",
                    f"TAG = '{tag}'\n",
                    "WORK_DIR = Path(f'/kaggle/working/{TAG}')\n",
                    "if WORK_DIR.exists():\n",
                    "    shutil.rmtree(WORK_DIR)\n",
                    "WORK_DIR.mkdir(parents=True, exist_ok=True)\n",
                    "os.chdir(WORK_DIR)\n",
                    "sys.path.insert(0, str(WORK_DIR))\n",
                    "\n",
                    "# Unpack base64 codebase hot-patch\n",
                    f"b64_patch = '{b64_patch}'\n",
                    "patch_bytes = base64.b64decode(b64_patch)\n",
                    "with zipfile.ZipFile(io.BytesIO(patch_bytes)) as z:\n",
                    "    z.extractall(WORK_DIR)\n",
                    "print(f'Extracted hot-patch codebase to {WORK_DIR}')\n",
                    "\n",
                    "# Locate AI-TOD Dataset\n",
                    "data_candidates = [\n",
                    "    Path('/kaggle/input/tiny-object-detection-in-aerial-images/AI-TOD'),\n",
                    "    Path('/kaggle/input/tiny-object-detection-in-aerial-images'),\n",
                    "    Path('/kaggle/input/aitoddatasets/AI-TOD'),\n",
                    "]\n",
                    "data_root = None\n",
                    "for dc in data_candidates:\n",
                    "    if dc.exists():\n",
                    "        data_root = dc\n",
                    "        break\n",
                    "if data_root is None:\n",
                    "    all_inputs = list(Path('/kaggle/input').rglob('*train.json'))\n",
                    "    if all_inputs:\n",
                    "        data_root = all_inputs[0].parents[1]\n",
                    "print(f'Resolved AI-TOD data root: {data_root}')\n",
                    "\n",
                    "# Install evaluator dependencies\n",
                    "subprocess.run(['pip', 'install', '-q', 'aitodpycocotools', 'torchmetrics', 'pycocotools'], capture_output=True)\n",
                    "\n",
                    f"base_cmd = '{cmd_str}'\n",
                    "full_cmd = f'{base_cmd} --data-root {data_root}'\n",
                    "print(f'Executing command: {full_cmd}')\n",
                    "proc = subprocess.run(full_cmd.split(), stdout=sys.stdout, stderr=sys.stderr)\n",
                    "print('Execution finished with returncode:', proc.returncode)\n",
                    "\n",
                    "# Copy output to working directory root for safe retrieval\n",
                    "out_dst = Path(f'/kaggle/working/tod_output')\n",
                    "if out_dst.exists(): shutil.rmtree(out_dst)\n",
                    "src_runs = WORK_DIR / 'runs'\n",
                    "if src_runs.exists():\n",
                    "    shutil.copytree(src_runs, out_dst / 'runs')\n",
                    "    print(f'Copied runs artifacts to {out_dst}')\n",
                    "\n",
                    "assert proc.returncode == 0\n"
                ]
            }
        ]

        nb = {
            "cells": nb_cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 2
        }
        (kdir / f"{tag}.ipynb").write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")

        # Set up auth profile
        profile = PROFILE_ROOT / account
        profile.mkdir(parents=True, exist_ok=True)
        shutil.copy(cred_file, profile / "kaggle.json")
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(profile)

        print(f"\n--> Pushing Kernel [{title}] to account [{account}]...")
        cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kdir)]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        print(f"  Stdout: {res.stdout.strip()}")
        if res.stderr.strip():
            print(f"  Stderr: {res.stderr.strip()}")

if __name__ == "__main__":
    main()
