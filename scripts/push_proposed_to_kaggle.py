"""
Push Proposed H-WIoU Sigma8 and Sigma10 and NWD to remaining Kaggle accounts with fresh unique slugs.
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
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_fair_profiles")
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
RUNTIME_DIR = ROOT / ".runtime/kaggle_aitod_fair"
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

TARGET_JOBS = [
    {
        "tag": "aitod_hwiou_sig8_prop_kgl",
        "slug": "tod-aitod-hwiou-sig8-prop-kgl",
        "title": "TOD AI-TOD H-WIoU Sigma8 Proposed Kaggle",
        "account": "amongus1504",
        "cred": CREDS_DIR / "kaggle.json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 8.0 --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag hwiou_sig8_prop_kgl",
    },
    {
        "tag": "aitod_hwiou_sig10_prop_kgl",
        "slug": "tod-aitod-hwiou-sig10-prop-kgl",
        "title": "TOD AI-TOD H-WIoU Sigma10 Proposed Kaggle",
        "account": "luongsythanh",
        "cred": CREDS_DIR / "kaggle (8).json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 10.0 --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag hwiou_sig10_prop_kgl",
    },
    {
        "tag": "aitod_nwd_prop_kgl",
        "slug": "tod-aitod-nwd-prop-kgl",
        "title": "TOD AI-TOD NWD Proposed Kaggle",
        "account": "hienquang06",
        "cred": CREDS_DIR / "kaggle (5).json",
        "cmd": "python scripts/train_frcnn_aitod.py --metric nwd --placement la_loss --box-loss metric --batch-size 4 --seed 42 --epochs 12 --eval-interval 3 --tag nwd_prop_kgl",
    },
]

def main():
    for exp in TARGET_JOBS:
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
            "machine_shape": "NvidiaTeslaT4"
        }
        (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

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
                    f"b64_patch = '{b64_patch}'\n",
                    "patch_bytes = base64.b64decode(b64_patch)\n",
                    "with zipfile.ZipFile(io.BytesIO(patch_bytes)) as z:\n",
                    "    z.extractall(WORK_DIR)\n",
                    "\n",
                    "data_candidates = [\n",
                    "    Path('/kaggle/input/tiny-object-detection-in-aerial-images/AI-TOD'),\n",
                    "    Path('/kaggle/input/tiny-object-detection-in-aerial-images'),\n",
                    "    Path('/kaggle/input/aitoddatasets/AI-TOD'),\n",
                    "]\n",
                    "data_root = next((d for d in data_candidates if d.exists()), Path('/kaggle/input'))\n",
                    "\n",
                    "subprocess.run(['pip', 'install', '-q', 'aitodpycocotools', 'torchmetrics', 'pycocotools'], capture_output=True)\n",
                    f"base_cmd = '{cmd_str}'\n",
                    "full_cmd = f'{base_cmd} --data-root {data_root}'\n",
                    "proc = subprocess.run(full_cmd.split(), stdout=sys.stdout, stderr=sys.stderr)\n",
                    "\n",
                    "out_dst = Path('/kaggle/working/tod_output')\n",
                    "if out_dst.exists(): shutil.rmtree(out_dst)\n",
                    "src_runs = WORK_DIR / 'runs'\n",
                    "if src_runs.exists(): shutil.copytree(src_runs, out_dst / 'runs')\n",
                    "assert proc.returncode == 0\n"
                ]
            }
        ]

        nb = {
            "cells": nb_cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
            },
            "nbformat": 4, "nbformat_minor": 2
        }
        (kdir / f"{tag}.ipynb").write_text(json.dumps(nb, indent=2), encoding="utf-8")

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
