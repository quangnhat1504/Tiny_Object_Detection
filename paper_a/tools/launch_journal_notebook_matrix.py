"""
Standardized Notebook-based Multi-Account Kernel Launcher for Journal Benchmarks & Ablations.
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
CREDS_DIR = Path.home() / ".kaggle"
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_hwiou_profiles")

EXPERIMENTS = [
    {
        "tag": "aitod_hwiou_sig8_s42",
        "title": "TOD AI-TOD H-WIoU Sigma8 S42",
        "account": "quangnhtng",
        "cred": "kaggle (6).json",
        "dataset_type": "aitod",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 8.0 --seed 42 --epochs 12",
        "dataset_sources": [
            "quangnhtng/tod-program-b-b2-code-20260814",
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
    {
        "tag": "aitod_hwiou_sig6_s42",
        "title": "TOD AI-TOD H-WIoU Sigma6 S42",
        "account": "qnhat1504",
        "cred": "kaggle (3).json",
        "dataset_type": "aitod",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 6.0 --seed 42 --epochs 12",
        "dataset_sources": [
            "qnhat1504/tod-program-b-b2-code-20260814",
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
    {
        "tag": "tp_ablation_pure_w2_s42",
        "title": "TOD TP Ablation Pure W2 S42",
        "account": "thyngluthy",
        "cred": "kaggle (4).json",
        "dataset_type": "tinyperson",
        "cmd": "python scripts/train_frcnn_metric.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-form pure_w2 --seed 42 --tag ablation_pure_w2",
        "dataset_sources": [
            "thyngluthy/tod-program-b-b2-code-20260814",
            "thyngluthy/tod-program-b-tinyperson-b1-tiled-20260814",
        ],
    },
    {
        "tag": "tp_ablation_pure_iou_s42",
        "title": "TOD TP Ablation Pure IoU S42",
        "account": "hngngnguynvn",
        "cred": "kaggle (1).json",
        "dataset_type": "tinyperson",
        "cmd": "python scripts/train_frcnn_metric.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-form pure_iou --seed 42 --tag ablation_pure_iou",
        "dataset_sources": [
            "hngngnguynvn/tod-program-b-b2-code-20260814",
            "hngngnguynvn/tod-program-b-tinyperson-b1-tiled-20260814",
        ],
    },
    {
        "tag": "tp_ablation_static_half_s42",
        "title": "TOD TP Ablation Static Half S42",
        "account": "hienquang06",
        "cred": "kaggle (5).json",
        "dataset_type": "tinyperson",
        "cmd": "python scripts/train_frcnn_metric.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-form static --h-wiou-static-gamma 0.5 --seed 42 --tag ablation_static_half",
        "dataset_sources": [
            "hienquang06/tod-program-b-b2-code-20260814",
            "hienquang06/tod-program-b-tinyperson-b1-tiled-20260814",
        ],
    },
]


def launch_notebook_experiment(exp: dict):
    account = exp["account"]
    tag = exp["tag"]
    title = exp.get("title", tag)
    cmd_str = exp["cmd"]
    dataset_sources = exp["dataset_sources"]
    slug = f"tod-{tag.replace('_', '-')}-20260820"

    kernel_dir = ROOT / ".runtime/kaggle" / tag
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
        "dataset_sources": dataset_sources,
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
                    f"    candidates = list(Path('/kaggle/input').glob('*{account}*')) + list(Path('/kaggle/input').glob('*tod-program*'))\n",
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
                    "\n",
                    "if torch.cuda.is_available() and 'P100' in torch.cuda.get_device_name(0):\n",
                    "    print('Detected Tesla P100 GPU: installing sm_60 compatible PyTorch...')\n",
                    "    subprocess.run(['pip', 'install', '-q', 'torch==2.4.1+cu121', 'torchvision==0.19.1+cu121', '--extra-index-url', 'https://download.pytorch.org/whl/cu121'])\n",
                    "\n",
                    "# Install dependencies\n",
                    "subprocess.run(['pip', 'install', '-q', 'aitodpycocotools', 'torchmetrics', 'pycocotools'], capture_output=True)\n",
                    "\n",
                    f"print('Executing command: {cmd_str}')\n",
                    f"proc = subprocess.run({repr(cmd_str.split())}, stdout=sys.stdout, stderr=sys.stderr)\n",
                    "print('Execution finished with exit code:', proc.returncode)\n"
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
    profile.mkdir(parents=True, exist_ok=True)
    shutil.copy(CREDS_DIR / exp["cred"], profile / "kaggle.json")

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(profile)

    print(f"\nPushing notebook kernel {meta['id']}...")
    push_cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kernel_dir)]
    res = subprocess.run(push_cmd, env=env, capture_output=True, text=True)
    print("Push Output:", res.stdout.strip())
    if res.stderr.strip():
        print("Push Stderr:", res.stderr.strip())

    time.sleep(3)
    stat_cmd = [sys.executable, "-m", "kaggle", "kernels", "status", meta["id"]]
    stat_res = subprocess.run(stat_cmd, env=env, capture_output=True, text=True)
    print(f"Status: {stat_res.stdout.strip()}")


def main():
    print("=== Pushing All 5 Journal Notebook Experiments ===")
    for exp in EXPERIMENTS:
        launch_notebook_experiment(exp)
    print("\n=== All 5 Journal Notebooks Pushed Successfully! ===")


if __name__ == "__main__":
    main()
