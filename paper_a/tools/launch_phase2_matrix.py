"""
Launch Phase 2 Expanded Experiment Matrix across 5 Kaggle accounts on GPU Tesla T4.
"""
from __future__ import annotations
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"

PHASE2_EXPERIMENTS = [
    {
        "stt": 1,
        "suite": "TinyPerson Multi-Seed",
        "tag": "tp_hwiou_sig8_s123",
        "slug": "tod-tp-hwiou-sig8-s123",
        "title": "tod-tp-hwiou-sig8-s123",
        "account": "hngtrngtn",
        "cred": "kaggle (7).json",
        "desc": "TinyPerson H-WIoU sigma_0=8.0px Seed 123",
        "cmd": "python -u scripts/train_frcnn_metric.py --metric h_wiou --placement la_loss --box-loss h_wiou --h-wiou-sigma-0 8.0 --seed 123 --tag hwiou_sig8_s123",
        "dataset_sources": [
            "hngtrngtn/tod-program-b-b2-code-20260814",
            "hngtrngtn/tod-program-b-tinyperson-b1-tiled-20260814",
        ],
    },
    {
        "stt": 2,
        "suite": "TinyPerson Multi-Seed",
        "tag": "tp_hwiou_sig8_s2024",
        "slug": "tod-tp-hwiou-sig8-s2024",
        "title": "tod-tp-hwiou-sig8-s2024",
        "account": "hngngnguynvn",
        "cred": "kaggle (4).json",
        "desc": "TinyPerson H-WIoU sigma_0=8.0px Seed 2024",
        "cmd": "python -u scripts/train_frcnn_metric.py --metric h_wiou --placement la_loss --box-loss h_wiou --h-wiou-sigma-0 8.0 --seed 2024 --tag hwiou_sig8_s2024",
        "dataset_sources": [
            "hngngnguynvn/tod-program-b-b2-code-20260814",
            "hngngnguynvn/tod-program-b-tinyperson-b1-tiled-20260814",
        ],
    },
    {
        "stt": 3,
        "suite": "AI-TOD SOTA Benchmark",
        "tag": "aitod_nwd_s42",
        "slug": "tod-aitod-nwd-s42",
        "title": "tod-aitod-nwd-s42",
        "account": "hienquang06",
        "cred": "kaggle (5).json",
        "desc": "AI-TOD-v2 NWD Metric (8 classes, seed 42)",
        "cmd": "python -u scripts/train_frcnn_aitod.py --metric nwd --placement la_loss --box-loss metric --batch-size 2 --seed 42 --epochs 12",
        "dataset_sources": [
            "hienquang06/tod-program-b-b2-code-20260814",
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
    {
        "stt": 4,
        "suite": "AI-TOD SOTA Benchmark",
        "tag": "aitod_igwd_s42",
        "slug": "tod-aitod-igwd-s42",
        "title": "tod-aitod-igwd-s42",
        "account": "luongsythanh",
        "cred": "kaggle (8).json",
        "desc": "AI-TOD-v2 IGWD Metric (8 classes, seed 42)",
        "cmd": "python -u scripts/train_frcnn_aitod.py --metric igwd --placement la_loss --box-loss metric --batch-size 2 --seed 42 --epochs 12",
        "dataset_sources": [
            "luongsythanh/tod-program-b-b2-code-20260814",
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
    {
        "stt": 5,
        "suite": "AI-TOD Baseline",
        "tag": "aitod_baseline_s42",
        "slug": "tod-aitod-baseline-chunked-s42",
        "title": "tod-aitod-baseline-chunked-s42",
        "account": "pptlyn11",
        "cred": "kaggle (9).json",
        "desc": "AI-TOD-v2 Faster R-CNN Chunked IoU Baseline (8 classes, seed 42)",
        "cmd": "python -u scripts/train_frcnn_aitod.py --metric standard --placement la_loss --box-loss smooth_l1 --batch-size 2 --seed 42 --epochs 12",
        "dataset_sources": [
            "pptlyn11/tod-program-b-b2-code-20260814",
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
    {
        "stt": 6,
        "suite": "AI-TOD-v2 Benchmark",
        "tag": "aitod_hwiou_sig8_s42",
        "slug": "tod-aitod-hwiou-sig8-s42",
        "title": "tod-aitod-hwiou-sig8-s42",
        "account": "amongus1504",
        "cred": "kaggle.json",
        "desc": "AI-TOD H-WIoU sigma_0=8.0px (8 classes, seed 42)",
        "cmd": "python -u scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 8.0 --batch-size 2 --seed 42 --epochs 12",
        "dataset_sources": [
            "amongus1504/tod-program-b-b2-code-20260814",
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
    {
        "stt": 7,
        "suite": "AI-TOD-v2 Benchmark",
        "tag": "aitod_hwiou_sig6_s42",
        "slug": "tod-aitod-hwiou-sig6-s42",
        "title": "tod-aitod-hwiou-sig6-s42",
        "account": "dipphmngc",
        "cred": "kaggle (11).json",
        "desc": "AI-TOD H-WIoU sigma_0=6.0px (8 classes, seed 42)",
        "cmd": "python -u scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 6.0 --batch-size 2 --seed 42 --epochs 12",
        "dataset_sources": [
            "dipphmngc/tod-program-b-b2-code-20260814",
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
    {
        "stt": 8,
        "suite": "AI-TOD-v2 Benchmark",
        "tag": "aitod_hwiou_sig10_s42",
        "slug": "tod-aitod-hwiou-sig10-s42",
        "title": "tod-aitod-hwiou-sig10-s42",
        "account": "phuc1806",
        "cred": "kaggle (12).json",
        "desc": "AI-TOD H-WIoU sigma_0=10.0px (8 classes, seed 42)",
        "cmd": "python -u scripts/train_frcnn_aitod.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 10.0 --batch-size 2 --seed 42 --epochs 12",
        "dataset_sources": [
            "phuc1806/tod-program-b-b2-code-20260814",
            "simplestzyp/tiny-object-detection-in-aerial-images",
        ],
    },
]

common_files = {
    f.name: f.read_text(encoding="utf-8")
    for f in (ROOT / "common").glob("*.py")
}
common_metrics_files = {
    f.name: f.read_text(encoding="utf-8")
    for f in (ROOT / "common/metrics").glob("*.py")
}
datasets_files = {
    f.name: f.read_text(encoding="utf-8")
    for f in (ROOT / "paper_a/datasets").glob("*.py")
}
evaluation_files = {
    f.name: f.read_text(encoding="utf-8")
    for f in (ROOT / "paper_a/evaluation").glob("*.py")
}
metric_code = (ROOT / "scripts/train_frcnn_metric.py").read_text(encoding="utf-8")
aitod_code = (ROOT / "scripts/train_frcnn_aitod.py").read_text(encoding="utf-8")

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
            {"cell_type": "markdown", "metadata": {}, "source": [f"# {title}\n", f"Automated execution: `{cmd_str}`\n"]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
                "import os, sys, shutil, subprocess, json, torch\n",
                "from pathlib import Path\n",
                "code_src = None\n",
                "for cand in list(Path('/kaggle/input').rglob('common')):\n",
                "    if cand.is_dir() and (cand.parent / 'scripts').is_dir():\n",
                "        code_src = cand.parent\n",
                "        break\n",
                "if code_src is None:\n",
                f"    code_src = Path('/kaggle/input/datasets/{account}/tod-program-b-b2-code-20260814')\n",
                "if not code_src.exists():\n",
                f"    code_candidates = [p for p in Path('/kaggle/input').glob('*code*') if p.is_dir()]\n",
                "    if code_candidates:\n",
                "        code_src = code_candidates[0]\n",
                f"work_dir = Path('/kaggle/working/{tag}')\n",
                "if work_dir.exists():\n",
                "    shutil.rmtree(work_dir)\n",
                "if code_src and code_src.exists():\n",
                "    shutil.copytree(code_src, work_dir)\n",
                "else:\n",
                "    work_dir.mkdir(parents=True, exist_ok=True)\n",
                f"os.chdir(work_dir)\n",
                "sys.path.insert(0, str(work_dir))\n",
                "# Embed all common, metrics, datasets, evaluation files directly\n",
                "Path('common/metrics').mkdir(parents=True, exist_ok=True)\n",
                "Path('paper_a/datasets').mkdir(parents=True, exist_ok=True)\n",
                "Path('paper_a/evaluation').mkdir(parents=True, exist_ok=True)\n",
                "Path('paper_a/__init__.py').write_text('', encoding='utf-8')\n",
                "Path('scripts').mkdir(parents=True, exist_ok=True)\n",
                f"common_dict = {repr(common_files)}\n",
                f"metrics_dict = {repr(common_metrics_files)}\n",
                f"datasets_dict = {repr(datasets_files)}\n",
                f"evaluation_dict = {repr(evaluation_files)}\n",
                "for fname, code in common_dict.items():\n",
                "    Path(f'common/{fname}').write_text(code, encoding='utf-8')\n",
                "for fname, code in metrics_dict.items():\n",
                "    Path(f'common/metrics/{fname}').write_text(code, encoding='utf-8')\n",
                "for fname, code in datasets_dict.items():\n",
                "    Path(f'paper_a/datasets/{fname}').write_text(code, encoding='utf-8')\n",
                "for fname, code in evaluation_dict.items():\n",
                "    Path(f'paper_a/evaluation/{fname}').write_text(code, encoding='utf-8')\n",
                f"Path('scripts/train_frcnn_metric.py').write_text({repr(metric_code)}, encoding='utf-8')\n",
                f"Path('scripts/train_frcnn_aitod.py').write_text({repr(aitod_code)}, encoding='utf-8')\n",
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

    print(f"Pushing notebook kernel {account}/{slug} on GPU Tesla T4...")
    res = subprocess.run(["kaggle", "kernels", "push", "-p", str(kernel_dir)], env=env, capture_output=True, text=True)
    print(f"Push Output: {res.stdout.strip() or res.stderr.strip()}")

    try:
        api = KaggleApi()
        api.authenticate()
        st = api.kernels_status(f"{account}/{slug}")
        print(f"Status: {account}/{slug} has status \"{getattr(st, 'status', 'unknown')}\"")
    except Exception as e:
        print(f"Status check note: {e}")

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    print("="*80)
    print("DEPLOYING PHASE 2 EXPERIMENT MATRIX (5 RUNS ON GPU TESLA T4)")
    print("="*80)
    for exp in PHASE2_EXPERIMENTS:
        print(f"\n[{exp['stt']}/5] Launching {exp['slug']} on account {exp['account']} ({exp['desc']})...")
        launch_single(exp)
    print("\nPhase 2 deployment complete.")

if __name__ == "__main__":
    main()
