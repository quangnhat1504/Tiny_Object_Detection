"""
Master Multi-Account Launcher for Direction 1: Multi-Detector & Cross-Architecture Generalization Cluster.
Dispatches 13 parallel GPU jobs across all 13 isolated Kaggle accounts.
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
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_gen_profiles")
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR = ROOT / ".runtime/kaggle_generalization_cluster"
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
    # --- Group A: TinyPerson Architecture & Multi-Seed (5 Accounts) ---
    {
        "account": "amongus1504",
        "cred": CREDS_DIR / "kaggle (2).json",
        "dataset_type": "tinyperson",
        "tag": "tp_rpn_cascade_baseline_s42",
        "slug": "tod-tp-rpn-cascade-baseline-s42",
        "title": "TOD TP RPN Cascade Baseline S42",
        "cmd": "python scripts/train_frcnn_metric.py --metric standard --placement everywhere --box-loss smooth_l1 --rpn-cascade --seed 42 --tag rpn_cascade_baseline_s42",
    },
    {
        "account": "dipphmngc",
        "cred": CREDS_DIR / "kaggle (11).json",
        "dataset_type": "tinyperson",
        "tag": "tp_rpn_cascade_nwd_s42",
        "slug": "tod-tp-rpn-cascade-nwd-s42",
        "title": "TOD TP RPN Cascade NWD S42",
        "cmd": "python scripts/train_frcnn_metric.py --metric nwd --placement la_loss --box-loss metric --rpn-cascade --seed 42 --tag rpn_cascade_nwd_s42",
    },
    {
        "account": "hienquang06",
        "cred": CREDS_DIR / "kaggle (5).json",
        "dataset_type": "tinyperson",
        "tag": "tp_rpn_cascade_hwiou_s42",
        "slug": "tod-tp-rpn-cascade-hwiou-s42",
        "title": "TOD TP RPN Cascade H-WIoU S42 Proposed",
        "cmd": "python scripts/train_frcnn_metric.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 8.0 --rpn-cascade --seed 42 --tag rpn_cascade_hwiou_s42",
    },
    {
        "account": "thyngluthy",
        "cred": CREDS_DIR / "kaggle (4).json",
        "dataset_type": "tinyperson",
        "tag": "tp_hwiou_sig8_s123",
        "slug": "tod-tp-hwiou-sig8-s123",
        "title": "TOD TP H-WIoU Sigma8 S123 Replication",
        "cmd": "python scripts/train_frcnn_metric.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 8.0 --seed 123 --tag hwiou_sig8_s123",
    },
    {
        "account": "trieuvo123",
        "cred": CREDS_DIR / "kaggle (10).json",
        "dataset_type": "tinyperson",
        "tag": "tp_hwiou_sig8_s2024",
        "slug": "tod-tp-hwiou-sig8-s2024",
        "title": "TOD TP H-WIoU Sigma8 S2024 Replication",
        "cmd": "python scripts/train_frcnn_metric.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 8.0 --seed 2024 --tag hwiou_sig8_s2024",
    },

    # --- Group B: AI-TOD-v2 Architecture & Multi-Seed Matrix (8 Accounts) ---
    {
        "account": "hngngnguynvn",
        "cred": CREDS_DIR / "kaggle (1).json",
        "dataset_type": "aitod",
        "tag": "aitod_rpn_cascade_baseline_s42",
        "slug": "tod-aitod-rpn-cascade-baseline-s42",
        "title": "TOD AI-TOD RPN Cascade Baseline S42",
        "cmd": "python scripts/train_frcnn_aitod.py --metric standard --placement iou_smooth_l1 --box-loss iou_smooth_l1 --rpn-cascade --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag aitod_rpn_cascade_baseline_s42",
    },
    {
        "account": "hngtrngtn",
        "cred": CREDS_DIR / "kaggle (7).json",
        "dataset_type": "aitod",
        "tag": "aitod_rpn_cascade_hwiou_sig6_s42",
        "slug": "tod-aitod-rpn-cascade-hwiou-sig6-s42",
        "title": "TOD AI-TOD RPN Cascade H-WIoU S42 Proposed",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 6.0 --rpn-cascade --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag aitod_rpn_cascade_hwiou_sig6_s42",
    },
    {
        "account": "luongsythanh",
        "cred": CREDS_DIR / "kaggle (8).json",
        "dataset_type": "aitod",
        "tag": "aitod_hwiou_sig6_s123",
        "slug": "tod-aitod-hwiou-sig6-s123",
        "title": "TOD AI-TOD H-WIoU Sigma6 S123 Replication",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 6.0 --batch-size 4 --seed 123 --epochs 12 --eval-interval 3 --tag hwiou_sig6_s123",
    },
    {
        "account": "ngquangnht",
        "cred": ROOT / ".runtime/kaggle/wp02/multi_account/cfg_ngquangnht/kaggle.json",
        "dataset_type": "aitod",
        "tag": "aitod_baseline_s123",
        "slug": "tod-aitod-baseline-s123",
        "title": "TOD AI-TOD Baseline S123 Replication",
        "cmd": "python scripts/train_frcnn_aitod.py --metric standard --placement iou_smooth_l1 --box-loss iou_smooth_l1 --batch-size 4 --seed 123 --epochs 12 --eval-interval 3 --tag baseline_s123",
    },
    {
        "account": "phuc1806",
        "cred": CREDS_DIR / "kaggle (12).json",
        "dataset_type": "aitod",
        "tag": "aitod_hwiou_sig6_s2024",
        "slug": "tod-aitod-hwiou-sig6-s2024",
        "title": "TOD AI-TOD H-WIoU Sigma6 S2024 Replication",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 6.0 --batch-size 4 --seed 2024 --epochs 12 --eval-interval 3 --tag hwiou_sig6_s2024",
    },
    {
        "account": "pptlyn11",
        "cred": CREDS_DIR / "kaggle (9).json",
        "dataset_type": "aitod",
        "tag": "aitod_baseline_s2024",
        "slug": "tod-aitod-baseline-s2024",
        "title": "TOD AI-TOD Baseline S2024 Replication",
        "cmd": "python scripts/train_frcnn_aitod.py --metric standard --placement iou_smooth_l1 --box-loss iou_smooth_l1 --batch-size 4 --seed 2024 --epochs 12 --eval-interval 3 --tag baseline_s2024",
    },
    {
        "account": "qnhat1504",
        "cred": CREDS_DIR / "kaggle (3).json",
        "dataset_type": "aitod",
        "tag": "aitod_nwd_s123",
        "slug": "tod-aitod-nwd-s123",
        "title": "TOD AI-TOD NWD S123 Replication",
        "cmd": "python scripts/train_frcnn_aitod.py --metric nwd --placement la_loss --box-loss metric --batch-size 4 --seed 123 --epochs 12 --eval-interval 3 --tag nwd_s123",
    },
    {
        "account": "quangnhtng",
        "cred": CREDS_DIR / "kaggle (6).json",
        "dataset_type": "aitod",
        "tag": "aitod_nwd_s2024",
        "slug": "tod-aitod-nwd-s2024",
        "title": "TOD AI-TOD NWD S2024 Replication",
        "cmd": "python scripts/train_frcnn_aitod.py --metric nwd --placement la_loss --box-loss metric --batch-size 4 --seed 2024 --epochs 12 --eval-interval 3 --tag nwd_s2024",
    },
]

def main():
    print("=" * 80)
    print("      LAUNCHING DIRECTION 1 GENERALIZATION MATRIX ON 13 KAGGLE GPU WORKERS      ")
    print("=" * 80)

    pushed = []
    skipped = []

    for job in JOBS:
        account = job["account"]
        tag = job["tag"]
        slug = job["slug"]
        title = job["title"]
        cmd_str = job["cmd"]
        cred_file = job["cred"]
        ds_type = job["dataset_type"]

        if not cred_file.exists():
            print(f"Skipping {account}: Cred file {cred_file} not found")
            skipped.append((account, "Cred missing"))
            continue

        profile = PROFILE_ROOT / account
        profile.mkdir(parents=True, exist_ok=True)
        shutil.copy(cred_file, profile / "kaggle.json")
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(profile)

        kdir = RUNTIME_DIR / tag
        kdir.mkdir(parents=True, exist_ok=True)

        if ds_type == "tinyperson":
            dataset_sources = [
                f"{account}/tod-program-b-b2-code-20260814",
                f"{account}/tod-program-b-tinyperson-b1-tiled-20260814",
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
                "source": [f"# {title}\n", f"Automated Execution: `{cmd_str}`\n"]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import os, sys, base64, io, zipfile, subprocess, shutil\n",
                    "from pathlib import Path\n",
                    "print('GPU Available:', subprocess.getoutput('nvidia-smi -L'))\n",
                    "\n",
                    "# 1. Unpack hot-patch code payload\n",
                    f"b64_data = '{b64_patch}'\n",
                    "buf = io.BytesIO(base64.b64decode(b64_data))\n",
                    "with zipfile.ZipFile(buf, 'r') as z:\n",
                    "    z.extractall('/kaggle/working')\n",
                    "print('Unpacked hot-patch files:', len(list(Path('/kaggle/working').rglob('*.py'))))\n",
                    "\n",
                    "# 2. Locate Dataset Root\n",
                    "data_root = None\n",
                    "input_p = Path('/kaggle/input')\n",
                    f"ds_mode = '{ds_type}'\n",
                    "if ds_mode == 'tinyperson':\n",
                    "    for p in input_p.rglob('mini_train_bundle.json'):\n",
                    "        data_root = p.parent\n",
                    "        break\n",
                    "    if data_root is None:\n",
                    "        for p in input_p.glob('*tinyperson*'):\n",
                    "            if p.is_dir():\n",
                    "                data_root = p\n",
                    "                break\n",
                    "else:\n",
                    "    for p in input_p.rglob('AI-TOD'):\n",
                    "        if p.is_dir():\n",
                    "            data_root = p\n",
                    "            break\n",
                    "    if data_root is None:\n",
                    "        for p in input_p.glob('*tiny-object*'):\n",
                    "            if p.is_dir():\n",
                    "                data_root = p\n",
                    "                break\n",
                    "print('Discovered Dataset Root:', data_root)\n",
                    "\n",
                    "# 3. Execute Training Command\n",
                    f"base_cmd = '{cmd_str}'\n",
                    "if ds_mode == 'tinyperson' and data_root:\n",
                    "    exec_cmd = f'{base_cmd} --data-root {data_root}'\n",
                    "elif ds_mode == 'aitod' and data_root:\n",
                    "    exec_cmd = f'{base_cmd} --data-root {data_root}'\n",
                    "else:\n",
                    "    exec_cmd = base_cmd\n",
                    "\n",
                    "print('Executing command:', exec_cmd)\n",
                    "sys.stdout.flush()\n",
                    "proc = subprocess.Popen(exec_cmd, shell=True, cwd='/kaggle/working', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)\n",
                    "for line in proc.stdout:\n",
                    "    print(line, end='')\n",
                    "proc.wait()\n",
                    "print('Process finished with exit code:', proc.returncode)\n",
                ]
            }
        ]

        nb = {
            "cells": nb_cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.10.12"}
            },
            "nbformat": 4,
            "nbformat_minor": 2
        }
        (kdir / f"{tag}.ipynb").write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")

        print(f"Pushing {slug} to {account}...")
        res = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kdir)], env=env, capture_output=True, text=True)
        out = (res.stdout + " " + res.stderr).strip()

        if "successfully pushed" in out:
            print(f"  [SUCCESS] {account:<15} -> {slug}")
            pushed.append((account, slug))
        else:
            print(f"  [ERROR]   {account:<15} -> {out[:80]}")
            skipped.append((account, out[:80]))

    print("\n" + "=" * 80)
    print(f"DISPATCH COMPLETED: {len(pushed)} Jobs Running on Kaggle GPU Cluster | {len(skipped)} Errors")
    print("=" * 80)

if __name__ == "__main__":
    main()
