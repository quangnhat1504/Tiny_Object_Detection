"""
Master Cloud GPU Launcher for the Exact 8-Experiment Journal & Benchmark Matrix.
Matches the user's exact specification table:
1. amongus1504 -> tod-aitod-baseline-s42       (AI-TOD-v2 Faster R-CNN Baseline 8 classes)
2. trieuvo123  -> tod-aitod-hwiou-sig8-s42     (AI-TOD-v2 H-WIoU sigma=8.0px 8 classes)
3. quangnhtng  -> tod-aitod-hwiou-sig6-s42     (AI-TOD-v2 H-WIoU sigma=6.0px 8 classes)
4. phuc1806    -> tod-aitod-hwiou-sig10-s42    (AI-TOD-v2 H-WIoU sigma=10.0px 8 classes)
5. thyngluthy  -> tod-tp-ablation-pure-w2-s42  (TinyPerson Pure W2 gamma=0)
6. hienquang06 -> tod-tp-ablation-pure-iou-s42 (TinyPerson Pure IoU gamma=1)
7. dipphmngc   -> tod-tp-ablation-static-half-s42 (TinyPerson Static Blend gamma=0.5)
8. hngngnguynvn-> tod-tp-ablation-exp-form-s42 (TinyPerson Exponential gamma_exp)
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
STAGE_DIR = Path(r"C:\tmp\tod_aitod_code_stage")
STAGE_DIR.mkdir(parents=True, exist_ok=True)

# Ensure fresh code in stage
for f in ["common/metrics/h_wiou.py", "common/model.py", "common/config.py", "common/dataset.py",
          "scripts/train_frcnn_aitod.py", "scripts/train_frcnn_metric.py"]:
    dst = STAGE_DIR / f
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / f, dst)

EXACT_8_EXPERIMENTS = [
    {
        "stt": 1,
        "dataset_type": "AI-TOD-v2",
        "tag": "aitod_baseline_s42",
        "slug": "tod-aitod-baseline-s42",
        "title": "tod-aitod-baseline-s42",
        "account": "amongus1504",
        "cred": "kaggle.json",
        "desc": "Faster R-CNN Baseline (8 classes)",
        "cmd": "python scripts/train_frcnn_aitod.py --metric standard --placement everywhere --box-loss smooth_l1 --batch-size 2 --seed 42 --epochs 12",
        "dataset_sources": [
            "amongus1504/tod-program-b-b2-code-20260814",
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
    {
        "stt": 2,
        "dataset_type": "AI-TOD-v2",
        "tag": "aitod_hwiou_sig8_s42",
        "slug": "tod-aitod-hwiou-sig8-s42",
        "title": "tod-aitod-hwiou-sig8-s42",
        "account": "qnhat1504",
        "cred": "kaggle (3).json",
        "desc": "H-WIoU (sigma_0 = 8.0 px, 8 classes)",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 8.0 --batch-size 2 --seed 42 --epochs 12",
        "dataset_sources": [
            "qnhat1504/tod-program-b-b2-code-20260814",
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
    {
        "stt": 3,
        "dataset_type": "AI-TOD-v2",
        "tag": "aitod_hwiou_sig6_s42",
        "slug": "tod-aitod-hwiou-sig6-s42",
        "title": "tod-aitod-hwiou-sig6-s42",
        "account": "quangnhtng",
        "cred": "kaggle (6).json",
        "desc": "H-WIoU (sigma_0 = 6.0 px, 8 classes)",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 6.0 --batch-size 2 --seed 42 --epochs 12",
        "dataset_sources": [
            "quangnhtng/tod-program-b-b2-code-20260814",
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
    {
        "stt": 4,
        "dataset_type": "AI-TOD-v2",
        "tag": "aitod_hwiou_sig10_s42",
        "slug": "tod-aitod-hwiou-sig10-s42",
        "title": "tod-aitod-hwiou-sig10-s42",
        "account": "phuc1806",
        "cred": "kaggle (12).json",
        "desc": "H-WIoU (sigma_0 = 10.0 px, 8 classes)",
        "cmd": "python scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 10.0 --batch-size 2 --seed 42 --epochs 12",
        "dataset_sources": [
            "phuc1806/tod-program-b-b2-code-20260814",
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
    {
        "stt": 5,
        "dataset_type": "TinyPerson",
        "tag": "tp_ablation_pure_w2_s42",
        "slug": "tod-tp-ablation-pure-w2-s42",
        "title": "tod-tp-ablation-pure-w2-s42",
        "account": "thyngluthy",
        "cred": "kaggle (4).json",
        "desc": "Phan tach Ablation Pure W2 (gamma = 0)",
        "cmd": "python scripts/train_frcnn_metric.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-form pure_w2 --seed 42 --tag ablation_pure_w2",
        "dataset_sources": [
            "thyngluthy/tod-program-b-b2-code-20260814",
            "thyngluthy/tod-program-b-tinyperson-b1-tiled-20260814",
        ],
    },
    {
        "stt": 6,
        "dataset_type": "TinyPerson",
        "tag": "tp_ablation_pure_iou_s42",
        "slug": "tod-tp-ablation-pure-iou-s42",
        "title": "tod-tp-ablation-pure-iou-s42",
        "account": "hienquang06",
        "cred": "kaggle (5).json",
        "desc": "Phan tach Ablation Pure IoU (gamma = 1)",
        "cmd": "python scripts/train_frcnn_metric.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-form pure_iou --seed 42 --tag ablation_pure_iou",
        "dataset_sources": [
            "hienquang06/tod-program-b-b2-code-20260814",
            "hienquang06/tod-program-b-tinyperson-b1-tiled-20260814",
        ],
    },
    {
        "stt": 7,
        "dataset_type": "TinyPerson",
        "tag": "tp_ablation_static_half_s42",
        "slug": "tod-tp-ablation-static-half-s42",
        "title": "tod-tp-ablation-static-half-s42",
        "account": "dipphmngc",
        "cred": "kaggle (11).json",
        "desc": "Phan tach Ablation Static Blend (gamma = 0.5)",
        "cmd": "python scripts/train_frcnn_metric.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-form static --h-wiou-static-gamma 0.5 --seed 42 --tag ablation_static_half",
        "dataset_sources": [
            "dipphmngc/tod-program-b-b2-code-20260814",
            "dipphmngc/tod-program-b-tinyperson-b1-tiled-20260814",
        ],
    },
    {
        "stt": 8,
        "dataset_type": "TinyPerson",
        "tag": "tp_ablation_exp_form_s42",
        "slug": "tod-tp-ablation-exp-form-s42",
        "title": "tod-tp-ablation-exp-form-s42",
        "account": "hngngnguynvn",
        "cred": "kaggle (1).json",
        "desc": "Khao sat dang ham Exponential gamma_exp",
        "cmd": "python scripts/train_frcnn_metric.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-form exponential --seed 42 --tag ablation_exp_form",
        "dataset_sources": [
            "hngngnguynvn/tod-program-b-b2-code-20260814",
            "hngngnguynvn/tod-program-b-tinyperson-b1-tiled-20260814",
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

    print(f"\n[{exp['stt']}/8] Syncing code dataset for {account}...")
    sync_cmd = [sys.executable, "-m", "kaggle", "datasets", "version", "-p", str(STAGE_DIR), "--dir-mode", "zip", "-m", "Sync full journal codebase"]
    subprocess.run(sync_cmd, env=env, capture_output=True)

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

    h_wiou_code = (ROOT / "common/metrics/h_wiou.py").read_text(encoding="utf-8")
    config_code = (ROOT / "common/config.py").read_text(encoding="utf-8")
    aitod_code = (ROOT / "scripts/train_frcnn_aitod.py").read_text(encoding="utf-8")
    coco_orig_code = (ROOT / "paper_a/datasets/coco_original.py").read_text(encoding="utf-8")
    iou_code = (ROOT / "common/metrics/iou.py").read_text(encoding="utf-8")
    metrics_init_code = (ROOT / "common/metrics/__init__.py").read_text(encoding="utf-8")

    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n", f"Automated execution: `{cmd_str}`\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "import os, sys, shutil, subprocess, json, torch\n",
                "from pathlib import Path\n",
                f"code_src = Path('/kaggle/input/datasets/{account}/tod-program-b-b2-code-20260814')\n",
                "if not code_src.exists():\n",
                f"    candidates = list(Path('/kaggle/input').glob('*{account}*')) + list(Path('/kaggle/input').glob('*tod-program*'))\n",
                "    if candidates:\n",
                "        code_src = candidates[0]\n",
                f"work_dir = Path('/kaggle/working/{tag}')\n",
                "if work_dir.exists():\n",
                "    shutil.rmtree(work_dir)\n",
                "if code_src.exists():\n",
                "    shutil.copytree(code_src, work_dir)\n",
                "else:\n",
                "    work_dir.mkdir(parents=True, exist_ok=True)\n",
                f"os.chdir(work_dir)\n",
                "sys.path.insert(0, str(work_dir))\n",
                "# Embed self-contained patched files directly\n",
                "Path('common/metrics').mkdir(parents=True, exist_ok=True)\n",
                "Path('paper_a/datasets').mkdir(parents=True, exist_ok=True)\n",
                f"Path('common/metrics/__init__.py').write_text({repr(metrics_init_code)}, encoding='utf-8')\n",
                f"Path('common/metrics/h_wiou.py').write_text({repr(h_wiou_code)}, encoding='utf-8')\n",
                f"Path('common/metrics/iou.py').write_text({repr(iou_code)}, encoding='utf-8')\n",
                f"Path('common/config.py').write_text({repr(config_code)}, encoding='utf-8')\n",
                f"Path('scripts/train_frcnn_aitod.py').write_text({repr(aitod_code)}, encoding='utf-8')\n",
                f"Path('paper_a/datasets/coco_original.py').write_text({repr(coco_orig_code)}, encoding='utf-8')\n",
                "torch_cache = work_dir / 'torch_cache'\n",
                "if torch_cache.exists():\n",
                "    dst_cache = Path.home() / '.cache/torch'\n",
                "    dst_cache.mkdir(parents=True, exist_ok=True)\n",
                "    for f in (torch_cache / 'hub/checkpoints').glob('*.pth'):\n",
                "        (dst_cache / 'hub/checkpoints').mkdir(parents=True, exist_ok=True)\n",
                "        shutil.copy(f, dst_cache / 'hub/checkpoints' / f.name)\n",
                "if torch.cuda.is_available() and 'P100' in torch.cuda.get_device_name(0):\n",
                "    print('Detected Tesla P100 GPU: installing sm_60 compatible PyTorch...')\n",
                "    subprocess.run(['pip', 'install', '-q', 'torch==2.4.1+cu121', 'torchvision==0.19.1+cu121', '--extra-index-url', 'https://download.pytorch.org/whl/cu121'])\n",
                "subprocess.run(['pip', 'install', '-q', 'aitodpycocotools', 'torchmetrics', 'pycocotools'], capture_output=True)\n",
                f"print('Executing command: {cmd_str}')\n",
                f"proc = subprocess.run({repr(cmd_str.split())}, stdout=sys.stdout, stderr=sys.stderr)\n",
                "print('Execution finished with exit code:', proc.returncode)\n"
            ]}
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 2
    }
    (kernel_dir / f"{tag}.ipynb").write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")

    print(f"Pushing notebook kernel {meta['id']}...")
    push_cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kernel_dir)]
    res = subprocess.run(push_cmd, env=env, capture_output=True, text=True)
    print("Push Output:", res.stdout.strip())
    time.sleep(2)
    stat_cmd = [sys.executable, "-m", "kaggle", "kernels", "status", meta["id"]]
    print("Status:", subprocess.run(stat_cmd, env=env, capture_output=True, text=True).stdout.strip())


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 115)
    print("         DEPLOYING EXACT 8-EXPERIMENT MATRIX ACROSS 8 ISOLATED KAGGLE ACCOUNTS          ")
    print("=" * 115)

    for exp in EXACT_8_EXPERIMENTS:
        launch_single(exp)

    print("\n" + "=" * 115)
    print("All 8 experiments successfully queued/running on Kaggle GPU!")


if __name__ == "__main__":
    main()
