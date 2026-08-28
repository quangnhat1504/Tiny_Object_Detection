"""
Launch Extended Journal Ablation Suite on available Kaggle accounts.
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

EXTENDED_ABLATIONS = [
    {
        "stt": 1,
        "dataset_type": "TinyPerson",
        "tag": "tp_ablation_sig4_s42",
        "slug": "tod-tp-ablation-sig4-s42",
        "title": "tod-tp-ablation-sig4-s42",
        "account": "hngtrngtn",
        "cred": "kaggle (7).json",
        "desc": "Sensitivity sigma_0 = 4.0 px",
        "cmd": "python scripts/train_frcnn_metric.py --metric h_wiou --placement h_wiou --box-loss h_wiou --h-wiou-sigma-0 4.0 --seed 42 --tag ablation_sig4",
        "dataset_sources": [
            "hngtrngtn/tod-program-b-b2-code-20260814",
            "hngtrngtn/tod-program-b-tinyperson-b1-tiled-20260814",
        ],
    },
    {
        "stt": 2,
        "dataset_type": "TinyPerson",
        "tag": "tp_ablation_sig12_s42",
        "slug": "tod-tp-ablation-sig12-s42",
        "title": "tod-tp-ablation-sig12-s42",
        "account": "luongsythanh",
        "cred": "kaggle (8).json",
        "desc": "Sensitivity sigma_0 = 12.0 px",
        "cmd": "python -u scripts/train_frcnn_metric.py --metric h_wiou --placement la_loss --box-loss h_wiou --h-wiou-sigma-0 12.0 --seed 42 --tag ablation_sig12",
        "dataset_sources": [
            "luongsythanh/tod-program-b-b2-code-20260814",
            "luongsythanh/tod-program-b-tinyperson-b1-tiled-20260814",
        ],
    },
    {
        "stt": 3,
        "dataset_type": "TinyPerson",
        "tag": "tp_ablation_form_sigmoid_s42",
        "slug": "tod-tp-ablation-form-sigmoid-s42",
        "title": "tod-tp-ablation-form-sigmoid-s42",
        "account": "pptlyn11",
        "cred": "kaggle (9).json",
        "desc": "Homotopy Form: Sigmoid tau=2.0",
        "cmd": "python -u scripts/train_frcnn_metric.py --metric h_wiou --placement la_loss --box-loss h_wiou --h-wiou-form sigmoid --seed 42 --tag ablation_form_sigmoid",
        "dataset_sources": [
            "pptlyn11/tod-program-b-b2-code-20260814",
            "pptlyn11/tod-program-b-tinyperson-b1-tiled-20260814",
        ],
    },
    {
        "stt": 4,
        "dataset_type": "TinyPerson",
        "tag": "tp_ablation_place_la_s42",
        "slug": "tod-tp-ablation-place-la-s42",
        "title": "tod-tp-ablation-place-la-s42",
        "account": "dipphmngc",
        "cred": "kaggle (11).json",
        "desc": "Module Placement: RPN Label Assignment Only",
        "cmd": "python -u scripts/train_frcnn_metric.py --metric h_wiou --placement la --box-loss smooth_l1 --seed 42 --tag ablation_place_la",
        "dataset_sources": [
            "dipphmngc/tod-program-b-b2-code-20260814",
            "dipphmngc/tod-program-b-tinyperson-b1-tiled-20260814",
        ],
    },
    {
        "stt": 5,
        "dataset_type": "TinyPerson",
        "tag": "tp_ablation_place_loss_s42",
        "slug": "tod-tp-ablation-place-loss-s42",
        "title": "tod-tp-ablation-place-loss-s42",
        "account": "hienquang06",
        "cred": "kaggle (5).json",
        "desc": "Module Placement: RoI Box Loss Only",
        "cmd": "python -u scripts/train_frcnn_metric.py --metric h_wiou --placement loss --box-loss h_wiou --seed 42 --tag ablation_place_loss",
        "dataset_sources": [
            "hienquang06/tod-program-b-b2-code-20260814",
            "hienquang06/tod-program-b-tinyperson-b1-tiled-20260814",
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
metric_code = (ROOT / "scripts/train_frcnn_metric.py").read_text(encoding="utf-8")

def launch_single(exp: dict):
    account = exp["account"]
    slug = exp["slug"]
    tag = exp["tag"]
    title = exp["title"]
    cmd_str = exp["cmd"]
    if not cmd_str.startswith("python -u"):
        cmd_str = cmd_str.replace("python ", "python -u ")
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
                "# Embed all common and common/metrics files directly\n",
                "Path('common/metrics').mkdir(parents=True, exist_ok=True)\n",
                "Path('scripts').mkdir(parents=True, exist_ok=True)\n",
                f"common_dict = {repr(common_files)}\n",
                f"metrics_dict = {repr(common_metrics_files)}\n",
                "for fname, code in common_dict.items():\n",
                "    Path(f'common/{fname}').write_text(code, encoding='utf-8')\n",
                "for fname, code in metrics_dict.items():\n",
                "    Path(f'common/metrics/{fname}').write_text(code, encoding='utf-8')\n",
                f"Path('scripts/train_frcnn_metric.py').write_text({repr(metric_code)}, encoding='utf-8')\n",
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
    print("=" * 95)
    print("       LAUNCHING EXTENDED JOURNAL ABLATION MATRIX ON FREE GPU T4 POOL       ")
    print("=" * 95)
    for exp in EXTENDED_ABLATIONS:
        print(f"\n[{exp['stt']}/{len(EXTENDED_ABLATIONS)}] Deploying {exp['desc']} ({exp['slug']}) on {exp['account']}...")
        launch_single(exp)

if __name__ == "__main__":
    main()
