import sys, os, subprocess, json, shutil, time
from pathlib import Path

ROOT = Path.cwd()
STAGE_DIR = ROOT / ".runtime/stage_official_tp_qnhat"
STAGE_DIR.mkdir(parents=True, exist_ok=True)
shutil.copytree(ROOT / "common", STAGE_DIR / "common", dirs_exist_ok=True)
shutil.copytree(ROOT / "scripts", STAGE_DIR / "scripts", dirs_exist_ok=True)
shutil.copytree(ROOT / "paper_a", STAGE_DIR / "paper_a", dirs_exist_ok=True)

# Also ensure torch_cache exists
torch_cache = ROOT / "torch_cache"
if torch_cache.exists():
    shutil.copytree(torch_cache, STAGE_DIR / "torch_cache", dirs_exist_ok=True)

account = "qnhat1504"
tag = "tp_official_standard_s42"
cmd_str = "python scripts/train_frcnn_metric.py --metric standard --placement everywhere --box-loss smooth_l1 --seed 42 --tag official_tp_standard_s42"

PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_official_tp_profiles")
profile = PROFILE_ROOT / account
env = os.environ.copy()
env["KAGGLE_CONFIG_DIR"] = str(profile)

meta_ds = {
    "title": f"Program B B2 Code Snapshot - {account}",
    "id": f"{account}/tod-program-b-b2-code-20260814",
    "licenses": [{"name": "other"}],
    "isPrivate": True,
}
(STAGE_DIR / "dataset-metadata.json").write_text(json.dumps(meta_ds, indent=2) + "\n", encoding="utf-8")

print(f"Uploading updated code package to {account}...")
subprocess.run(["kaggle", "datasets", "version", "-p", str(STAGE_DIR), "--dir-mode", "zip", "-m", "Add standard baseline update"], env=env)

time.sleep(5)

slug = f"tp-official-standard-s42-fair20"
k_dir = ROOT / ".runtime/local/program_b" / f"kaggle_qnhat_{tag}"
k_dir.mkdir(parents=True, exist_ok=True)

meta_k = {
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
(k_dir / "kernel-metadata.json").write_text(json.dumps(meta_k, indent=2) + "\n", encoding="utf-8")

script_cell = f"""
import subprocess, os, sys, shutil
from pathlib import Path

print("=== STARTING OFFICIAL TINYPERSON STANDARD BASELINE TRAINING ON TESLA T4 ===")
import torch
print(f"GPU: {{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}}")

work_dir = Path("/kaggle/working/tp_official_standard_s42")
work_dir.mkdir(parents=True, exist_ok=True)

input_dir = Path("/kaggle/input")
print("Available input directories:")
for p in input_dir.iterdir():
    print("  ->", p.name)

# Recursively locate code directory containing scripts/train_frcnn_metric.py
code_dir = None
for p in Path("/kaggle/input").rglob("train_frcnn_metric.py"):
    # If p is /kaggle/input/.../scripts/train_frcnn_metric.py, the root is p.parent.parent
    code_dir = p.parent.parent
    break

if code_dir is None:
    # Print full tree to diagnose
    print("Full input tree:")
    for p in Path("/kaggle/input").rglob("*"):
        print(" ", p)
    raise RuntimeError("Could not find train_frcnn_metric.py in /kaggle/input")

print(f"Found code directory at: {{code_dir}}")
for item in code_dir.iterdir():
    dst = work_dir / item.name
    if item.is_dir():
        shutil.copytree(item, dst, dirs_exist_ok=True)
    else:
        shutil.copy(item, dst)

os.chdir(str(work_dir))

# Setup torch cache for resnet50 pretrained weights
cache_src = work_dir / "torch_cache"
if cache_src.exists():
    os.environ["TORCH_HOME"] = str(work_dir / "torch_cache")

# Recursively locate dataset directory containing annotations
data_dir = None
for p in Path("/kaggle/input").rglob("*tiny_set_train_all.json"):
    # If p is /kaggle/input/.../annotations/mini_annotations/tiny_set_train_all.json
    # data_root is the parent of annotations
    data_dir = p.parent.parent.parent
    break

if data_dir is None:
    # Fallback to search for mini_annotations directory
    for p in Path("/kaggle/input").rglob("mini_annotations"):
        data_dir = p.parent.parent
        break

if data_dir is None:
    raise RuntimeError("Could not find TinyPerson dataset with annotations in /kaggle/input")

print(f"Found dataset directory at: {{data_dir}}")

# Execute training command
cmd = "{cmd_str} --data-root " + str(data_dir)
print(f"Running command: {{cmd}}")
res = subprocess.run(cmd, shell=True, text=True)
print(f"Training completed with returncode: {{res.returncode}}")
"""

nb = {
    "cells": [
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": script_cell.splitlines(keepends=True)
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}
(k_dir / f"{tag}.ipynb").write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")

print(f"Pushing kernel {account}/{slug} to Kaggle...")
res = subprocess.run(["kaggle", "kernels", "push", "-p", str(k_dir)], env=env, capture_output=True, text=True)
print(res.stdout)
print(res.stderr)
