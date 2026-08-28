"""
Launch remaining 5 Kaggle GPU accounts to complete the 12-account SOTA & Ablation Matrix.
Embeds all metric modules (alw, sa_alw, sa_alw_canonical, nwd, igwd, h_wiou, iou) to avoid missing import errors.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
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

REMAINING_5_EXPERIMENTS = [
    {
        "stt": 8,
        "tag": "aitod_cascade_s42",
        "slug": "tod-aitod-cascade-s42-20260824",
        "title": "tod-aitod-cascade-s42-20260824",
        "account": "hngtrngtn",
        "cred": "kaggle (7).json",
        "desc": "Cascade R-CNN (CVPR 2018, S42)",
        "cmd": "python scripts/train_frcnn_aitod.py --metric standard --placement everywhere --box-loss smooth_l1 --batch-size 2 --seed 42 --epochs 12 --tag cascade_rcnn",
        "dataset_sources": [
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
    {
        "stt": 9,
        "tag": "aitod_dotd_s42",
        "slug": "tod-aitod-dotd-s42-20260824",
        "title": "tod-aitod-dotd-s42-20260824",
        "account": "luongsythanh",
        "cred": "kaggle (8).json",
        "desc": "DotD (ICCV 2021, S42)",
        "cmd": "python scripts/train_frcnn_aitod.py --metric standard --placement la --box-loss metric --batch-size 2 --seed 42 --epochs 12 --tag dotd",
        "dataset_sources": [
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
    {
        "stt": 10,
        "tag": "aitod_simd_s42",
        "slug": "tod-aitod-simd-s42-20260824",
        "title": "tod-aitod-simd-s42-20260824",
        "account": "pptlyn11",
        "cred": "kaggle (9).json",
        "desc": "SimD (CVPR 2023, S42)",
        "cmd": "python scripts/train_frcnn_aitod.py --metric standard --placement la --box-loss metric --batch-size 2 --seed 42 --epochs 12 --tag simd",
        "dataset_sources": [
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
    {
        "stt": 11,
        "tag": "aitod_safit_s42",
        "slug": "tod-aitod-safit-s42-20260824",
        "title": "tod-aitod-safit-s42-20260824",
        "account": "trieuvo123",
        "cred": "kaggle (10).json",
        "desc": "SAFit (AAAI 2024, S42)",
        "cmd": "python scripts/train_frcnn_aitod.py --metric sa_alw_canonical --placement everywhere --box-loss metric --batch-size 2 --seed 42 --epochs 12 --tag safit",
        "dataset_sources": [
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
    {
        "stt": 12,
        "tag": "aitod_hwiou_cascade_s42",
        "slug": "tod-aitod-hwiou-cascade-s42-20260824",
        "title": "tod-aitod-hwiou-cascade-s42-20260824",
        "account": "phuc1806",
        "cred": "kaggle (12).json",
        "desc": "H-WIoU + Cascade Hybrid (S42)",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 8.0 --batch-size 2 --seed 42 --epochs 12 --tag hwiou_cascade",
        "dataset_sources": [
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
]


def launch_single(exp: dict):
    account = exp["account"]
    slug = exp["slug"]
    tag = exp["tag"]
    title = exp["title"]
    cmd_str = exp["cmd"]
    dataset_sources = exp["dataset_sources"]
    cred_file = exp["cred"]

    creds = json.loads((CREDS_DIR / cred_file).read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = creds["username"]
    env["KAGGLE_KEY"] = creds["key"]

    print(f"\n[{exp['stt']}/12] Preparing notebook for {account} ({slug})...")

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

    # Read all source modules
    aitod_adapter_code = (ROOT / "paper_a/datasets/aitodv2_adapter.py").read_text(encoding="utf-8")
    coco_orig_code = (ROOT / "paper_a/datasets/coco_original.py").read_text(encoding="utf-8")
    datasets_init_code = (ROOT / "paper_a/datasets/__init__.py").read_text(encoding="utf-8") if (ROOT / "paper_a/datasets/__init__.py").exists() else ""
    paper_a_init_code = (ROOT / "paper_a/__init__.py").read_text(encoding="utf-8") if (ROOT / "paper_a/__init__.py").exists() else ""

    # Read all evaluation modules
    eval_init_code = (ROOT / "paper_a/evaluation/__init__.py").read_text(encoding="utf-8") if (ROOT / "paper_a/evaluation/__init__.py").exists() else ""
    eval_aitod_code = (ROOT / "paper_a/evaluation/aitodv2_official.py").read_text(encoding="utf-8") if (ROOT / "paper_a/evaluation/aitodv2_official.py").exists() else ""
    eval_coco_code = (ROOT / "paper_a/evaluation/coco_contract.py").read_text(encoding="utf-8") if (ROOT / "paper_a/evaluation/coco_contract.py").exists() else ""

    h_wiou_code = (ROOT / "common/metrics/h_wiou.py").read_text(encoding="utf-8")
    alw_code = (ROOT / "common/metrics/alw.py").read_text(encoding="utf-8")
    sa_alw_code = (ROOT / "common/metrics/sa_alw.py").read_text(encoding="utf-8")
    sa_alw_can_code = (ROOT / "common/metrics/sa_alw_canonical.py").read_text(encoding="utf-8")
    nwd_code = (ROOT / "common/metrics/nwd.py").read_text(encoding="utf-8")
    igwd_code = (ROOT / "common/metrics/igwd.py").read_text(encoding="utf-8")
    iou_code = (ROOT / "common/metrics/iou.py").read_text(encoding="utf-8")
    metrics_init_code = (ROOT / "common/metrics/__init__.py").read_text(encoding="utf-8")

    config_code = (ROOT / "common/config.py").read_text(encoding="utf-8")
    model_code = (ROOT / "common/model.py").read_text(encoding="utf-8")
    dataset_code = (ROOT / "common/dataset.py").read_text(encoding="utf-8")
    train_utils_code = (ROOT / "common/train_utils.py").read_text(encoding="utf-8")
    cascade_code = (ROOT / "common/cascade.py").read_text(encoding="utf-8")
    assigner_code = (ROOT / "common/assigner.py").read_text(encoding="utf-8")
    aitod_code = (ROOT / "scripts/train_frcnn_aitod.py").read_text(encoding="utf-8")

    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n", f"Automated execution: `{cmd_str}`\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "import os, sys, shutil, subprocess, json, torch\n",
                "from pathlib import Path\n",
                f"work_dir = Path('/kaggle/working/{tag}')\n",
                "work_dir.mkdir(parents=True, exist_ok=True)\n",
                f"os.chdir(work_dir)\n",
                "sys.path.insert(0, str(work_dir))\n",
                "Path('common/metrics').mkdir(parents=True, exist_ok=True)\n",
                "Path('paper_a/datasets').mkdir(parents=True, exist_ok=True)\n",
                "Path('paper_a/evaluation').mkdir(parents=True, exist_ok=True)\n",
                "Path('scripts').mkdir(parents=True, exist_ok=True)\n",
                f"Path('paper_a/__init__.py').write_text({repr(paper_a_init_code)}, encoding='utf-8')\n",
                f"Path('paper_a/datasets/__init__.py').write_text({repr(datasets_init_code)}, encoding='utf-8')\n",
                f"Path('paper_a/datasets/aitodv2_adapter.py').write_text({repr(aitod_adapter_code)}, encoding='utf-8')\n",
                f"Path('paper_a/datasets/coco_original.py').write_text({repr(coco_orig_code)}, encoding='utf-8')\n",
                f"Path('paper_a/evaluation/__init__.py').write_text({repr(eval_init_code)}, encoding='utf-8')\n",
                f"Path('paper_a/evaluation/aitodv2_official.py').write_text({repr(eval_aitod_code)}, encoding='utf-8')\n",
                f"Path('paper_a/evaluation/coco_contract.py').write_text({repr(eval_coco_code)}, encoding='utf-8')\n",
                f"Path('common/metrics/__init__.py').write_text({repr(metrics_init_code)}, encoding='utf-8')\n",
                f"Path('common/metrics/h_wiou.py').write_text({repr(h_wiou_code)}, encoding='utf-8')\n",
                f"Path('common/metrics/alw.py').write_text({repr(alw_code)}, encoding='utf-8')\n",
                f"Path('common/metrics/sa_alw.py').write_text({repr(sa_alw_code)}, encoding='utf-8')\n",
                f"Path('common/metrics/sa_alw_canonical.py').write_text({repr(sa_alw_can_code)}, encoding='utf-8')\n",
                f"Path('common/metrics/nwd.py').write_text({repr(nwd_code)}, encoding='utf-8')\n",
                f"Path('common/metrics/igwd.py').write_text({repr(igwd_code)}, encoding='utf-8')\n",
                f"Path('common/metrics/iou.py').write_text({repr(iou_code)}, encoding='utf-8')\n",
                f"Path('common/config.py').write_text({repr(config_code)}, encoding='utf-8')\n",
                f"Path('common/model.py').write_text({repr(model_code)}, encoding='utf-8')\n",
                f"Path('common/dataset.py').write_text({repr(dataset_code)}, encoding='utf-8')\n",
                f"Path('common/train_utils.py').write_text({repr(train_utils_code)}, encoding='utf-8')\n",
                f"Path('common/cascade.py').write_text({repr(cascade_code)}, encoding='utf-8')\n",
                f"Path('common/assigner.py').write_text({repr(assigner_code)}, encoding='utf-8')\n",
                f"Path('scripts/train_frcnn_aitod.py').write_text({repr(aitod_code)}, encoding='utf-8')\n",
                "# Locate dataset\n",
                "print('=== Locating AI-TOD-v2 dataset ===')\n",
                "data_root = None\n",
                "for p in [Path('/kaggle/input/tiny-object-detection-in-aerial-images'), Path('/kaggle/input/ai-tod-v2'), Path('/kaggle/input')]:\n",
                "    if p.exists() and (p / 'train').exists():\n",
                "        data_root = p\n",
                "        break\n",
                "    for sub in p.glob('*ai-tod*'):\n",
                "        if (sub / 'train').exists():\n",
                "            data_root = sub\n",
                "            break\n",
                "print('Data root detected:', data_root)\n",
                f"cmd = {repr(cmd_str)} + (' --data-root ' + str(data_root) if data_root else '')\n",
                "print('Executing command:', cmd)\n",
                "subprocess.run(cmd, shell=True, check=True)\n",
                "print('=== Completed Training! ===')\n",
            ]}
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.10.12"}
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    (kernel_dir / f"{tag}.ipynb").write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")

    print(f"Pushing kernel {account}/{slug} to Kaggle...")
    res = subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], capture_output=True, text=True, env=env)
    print(f"Result: {res.stdout.strip() if res.stdout else res.stderr.strip()}")


def main():
    print("=== Launching 5 Remaining Kaggle GPU Accounts for SOTA Matrix ===")
    for exp in REMAINING_5_EXPERIMENTS:
        launch_single(exp)
        time.sleep(2)
    print("\n[SUCCESS] All 5 additional Kaggle accounts launched successfully!")


if __name__ == "__main__":
    main()
