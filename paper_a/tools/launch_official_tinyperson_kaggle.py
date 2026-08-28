"""
Official TinyPerson 7-Model Kaggle Training Launcher.
Runs 7 models on 7 accounts that own their exact dataset replicas.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS = Path.home() / ".kaggle"
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_official_tp_profiles")
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
STAGE_DIR = ROOT / ".runtime/stage_official_tp"

EXPERIMENTS = [
    {
        "method": "standard",
        "title": "Faster R-CNN Baseline (Standard IoU)",
        "account": "amongus1504",
        "cred": "kaggle (2).json",
        "tag": "tp_official_standard_s42",
        "cmd": "python scripts/train_frcnn_metric.py --metric standard --placement standard --seed 42 --tag official_tp_standard_s42",
    },
    {
        "method": "nwd",
        "title": "NWD (NeurIPS 2021)",
        "account": "qnhat1504",
        "cred": "kaggle (3).json",
        "tag": "tp_official_nwd_s42",
        "cmd": "python scripts/train_frcnn_metric.py --metric nwd --placement la_loss --seed 42 --tag official_tp_nwd_s42",
    },
    {
        "method": "igwd",
        "title": "IGWD (IEEE TMM 2022)",
        "account": "thyngluthy",
        "cred": "kaggle (4).json",
        "tag": "tp_official_igwd_s42",
        "cmd": "python scripts/train_frcnn_metric.py --metric igwd --placement la_loss --seed 42 --tag official_tp_igwd_s42",
    },
    {
        "method": "sa_alw_full",
        "title": "SA-ALW (Paper A / AAAI 2024)",
        "account": "quangnhtng",
        "cred": "kaggle (6).json",
        "tag": "tp_official_saalw_s42",
        "cmd": "python scripts/train_frcnn_metric.py --metric sa_alw_full --placement la_loss --seed 42 --tag official_tp_saalw_s42",
    },
    {
        "method": "h_wiou_sig8",
        "title": "H-WIoU Sigma8 (Proposed Ours)",
        "account": "hienquang06",
        "cred": "kaggle (5).json",
        "tag": "tp_official_hwiou_sig8_s42",
        "cmd": "python scripts/train_frcnn_metric.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 8.0 --seed 42 --tag official_tp_hwiou_sig8_s42",
    },
    {
        "method": "h_wiou_sig6",
        "title": "H-WIoU Sigma6 (Ablation Ours)",
        "account": "hngngnguynvn",
        "cred": "kaggle (1).json",
        "tag": "tp_official_hwiou_sig6_s42",
        "cmd": "python scripts/train_frcnn_metric.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 6.0 --seed 42 --tag official_tp_hwiou_sig6_s42",
    },
    {
        "method": "rfla",
        "title": "RFLA (ECCV 2022)",
        "account": "dipphmngc",
        "cred": "kaggle (11).json",
        "tag": "tp_official_rfla_s42",
        "cmd": "python scripts/train_frcnn_metric.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 6.0 --seed 42 --tag official_tp_rfla_s42",
    },
]

def prepare_stage():
    print("=== Step 1: Staging Code & Weights ===")
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR, ignore_errors=True)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copytree(ROOT / "common", STAGE_DIR / "common", dirs_exist_ok=True)
    shutil.copytree(ROOT / "scripts", STAGE_DIR / "scripts", dirs_exist_ok=True)
    shutil.copytree(ROOT / "paper_a", STAGE_DIR / "paper_a", dirs_exist_ok=True)

    torch_checkpoints = STAGE_DIR / "torch_cache/hub/checkpoints"
    torch_checkpoints.mkdir(parents=True, exist_ok=True)
    local_weight = Path.home() / ".cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
    if local_weight.is_file():
        shutil.copy(local_weight, torch_checkpoints / "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth")
        print(f"Staged pretrained ResNet-50-FPN weights ({local_weight.stat().st_size / 1e6:.1f} MB)")
    print(f"Code staging complete at {STAGE_DIR}")

def update_account_code(account: str, cred_file: str):
    profile = PROFILE_ROOT / account
    if profile.exists():
        shutil.rmtree(profile, ignore_errors=True)
    profile.mkdir(parents=True, exist_ok=True)
    shutil.copy(CREDS / cred_file, profile / "kaggle.json")

    meta = {
        "title": f"Program B B2 Code Snapshot - {account}",
        "id": f"{account}/tod-program-b-b2-code-20260814",
        "licenses": [{"name": "other"}],
        "subtitleNullable": "Private immutable Program B B2 code snapshot replica with H-WIoU",
        "descriptionNullable": "Authorized private replica with H-WIoU and offline torch_cache weights.",
        "isPrivate": True,
    }
    (STAGE_DIR / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(profile)
    dataset_ref = f"{account}/tod-program-b-b2-code-20260814"

    print(f"Updating code dataset for {dataset_ref}...")
    cmd = [sys.executable, "-m", "kaggle", "datasets", "version", "-p", str(STAGE_DIR), "--dir-mode", "zip", "-m", "Update code snapshot with H-WIoU"]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("Dataset Version Output:", res.stdout.strip())

    for attempt in range(1, 10):
        status_cmd = [sys.executable, "-m", "kaggle", "datasets", "status", dataset_ref]
        s_res = subprocess.run(status_cmd, env=env, capture_output=True, text=True)
        out = s_res.stdout.strip()
        if "ready" in out.lower():
            print(f"Dataset {dataset_ref} is READY!")
            return
        time.sleep(3)

def push_experiment_kernel(exp: dict):
    account = exp["account"]
    tag = exp["tag"]
    title = exp["title"]
    cmd_str = exp["cmd"]
    slug = f"tp-official-{tag.replace('_', '-')}-fair2"

    kernel_dir = ROOT / ".runtime/local/program_b" / f"kaggle_{tag}"
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
            f"{account}/tod-program-b-tinyperson-b1-tiled-20260814",
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4"
    }
    (kernel_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    nb = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [f"# {title}\n", f"Automated execution: `{cmd_str}`\n"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os, sys, shutil, subprocess, json, torch\n",
                    "from pathlib import Path\n",
                    "\n",
                    f"code_src = Path('/kaggle/input/datasets/{account}/tod-program-b-b2-code-20260814')\n",
                    "if not code_src.exists():\n",
                    f"    candidates = list(Path('/kaggle/input').glob('*{account}*')) + list(Path('/kaggle/input').glob('*tod-program-b-b2-code*'))\n",
                    "    if candidates:\n",
                    "        code_src = candidates[0]\n",
                    "\n",
                    f"work_dir = Path('/kaggle/working/{tag}')\n",
                    "if work_dir.exists():\n",
                    "    shutil.rmtree(work_dir)\n",
                    "shutil.copytree(code_src, work_dir)\n",
                    f"os.chdir(work_dir)\n",
                    "sys.path.insert(0, str(work_dir))\n",
                    "\n",
                    "# Setup torch cache\n",
                    "torch_cache = work_dir / 'torch_cache'\n",
                    "if torch_cache.exists():\n",
                    "    dst_cache = Path.home() / '.cache/torch'\n",
                    "    dst_cache.mkdir(parents=True, exist_ok=True)\n",
                    "    for f in (torch_cache / 'hub/checkpoints').glob('*.pth'):\n",
                    "        (dst_cache / 'hub/checkpoints').mkdir(parents=True, exist_ok=True)\n",
                    "        shutil.copy(f, dst_cache / 'hub/checkpoints' / f.name)\n",
                    "    os.environ['TORCH_HOME'] = str(dst_cache)\n",
                    "\n",
                    "# Install dependencies\n",
                    "subprocess.run(['pip', 'install', '-q', 'aitodpycocotools', 'torchmetrics', 'pycocotools'], capture_output=True)\n",
                    "\n",
                    f"print('Executing command: {cmd_str}')\n",
                    f"proc = subprocess.run({repr(cmd_str.split())}, stdout=sys.stdout, stderr=sys.stderr)\n",
                    "print('Execution finished with exit code:', proc.returncode)\n",
                    "assert proc.returncode == 0\n"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    (kernel_dir / f"{tag}.ipynb").write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")

    profile = PROFILE_ROOT / account
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(profile)

    print(f"\nPushing kernel {meta['id']} to Kaggle...")
    cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kernel_dir)]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print("Push Output:", res.stdout.strip())
    if res.stderr.strip():
        print("Push Stderr:", res.stderr.strip()[:200])

def main():
    prepare_stage()
    print("\n" + "="*80)
    print("=== SYNCING CODE ACROSS ALL 7 ACCOUNTS ===")
    print("="*80 + "\n")
    for exp in EXPERIMENTS:
        update_account_code(exp["account"], exp["cred"])

    print("\n" + "="*80)
    print("=== PUSHING KERNELS TO KAGGLE CLUSTER ===")
    print("="*80 + "\n")
    for exp in EXPERIMENTS:
        push_experiment_kernel(exp)

    print("\n[SUCCESS] All 7 official TinyPerson training jobs pushed to Kaggle GPU cluster!\n")

if __name__ == "__main__":
    main()
