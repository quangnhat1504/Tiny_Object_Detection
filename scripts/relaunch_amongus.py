"""
Master Launcher for Cascade Multi-Stage Homotopy & RFLA on Kaggle GPU Cluster (Tesla T4).
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
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_cascade_profiles")
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR = ROOT / ".runtime/kaggle_aitod_cascade"
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

CASCADE_EXPS = [
    {
        "tag": "aitod_cascade_baseline_s42",
        "slug": "tod-cascade-baseline-s42",
        "title": "tod-cascade-baseline-s42",
        "account": "amongus1504",
        "cred": CREDS_DIR / "kaggle (2).json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric standard --placement la_loss --box-loss smooth_l1 --batch-size 4 --seed 42 --epochs 12 --rpn-cascade --tag cascade_baseline_s42",
    },
]


def main():
    print("=" * 80)
    print("   RE-LAUNCHING FIXED CASCADE BASELINE ON AMONGUS1504   ")
    print("=" * 80)

    for exp in CASCADE_EXPS:
        account = exp["account"]
        tag = exp["tag"]
        slug = exp["slug"]
        title = exp["title"]
        cmd_str = exp["cmd"]
        cred_file = exp["cred"]

        kdir = RUNTIME_DIR / tag
        kdir.mkdir(parents=True, exist_ok=True)

        user_profile = PROFILE_ROOT / account
        user_profile.mkdir(parents=True, exist_ok=True)
        shutil.copy(cred_file, user_profile / "kaggle.json")

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
                "simplestzyp/tiny-object-detection-in-aerial-images"
            ],
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
                    "if WORK_DIR.exists(): shutil.rmtree(WORK_DIR)\n",
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
                "language_info": {"name": "python", "version": "3.10"},
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
            },
            "nbformat": 4,
            "nbformat_minor": 5
        }
        (kdir / f"{tag}.ipynb").write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")

        print(f"\n[Pushing to Kaggle Account: {account}] -> {slug}")
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(user_profile)

        push_cmd = ["kaggle", "kernels", "push", "-p", str(kdir)]
        p = subprocess.run(push_cmd, env=env, capture_output=True, text=True)
        print("STDOUT:", p.stdout.strip())
        if p.stderr.strip():
            print("STDERR:", p.stderr.strip())

    print("\n" + "=" * 80)
    print("   AMONGUS1504 RE-LAUNCHED SUCCESSFULLY!   ")
    print("=" * 80)


if __name__ == "__main__":
    main()
