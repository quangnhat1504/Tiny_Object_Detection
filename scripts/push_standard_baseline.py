import sys, os, subprocess, json
from pathlib import Path

PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_official_tp_profiles")
CREDS = Path.home() / ".kaggle"
ROOT = Path.cwd()

account = "amongus1504"
cred = "kaggle (2).json"
tag = "tp_official_standard_s42"
cmd_str = "python scripts/train_frcnn_metric.py --metric standard --placement la --box-loss smooth_l1 --seed 42 --tag official_tp_standard_s42"

profile = PROFILE_ROOT / account
profile.mkdir(parents=True, exist_ok=True)
import shutil
shutil.copy(CREDS / cred, profile / "kaggle.json")
env = os.environ.copy()
env["KAGGLE_CONFIG_DIR"] = str(profile)

slug = f"tp-official-{tag.replace('_', '-')}-fair2"
k_dir = ROOT / ".runtime/local/program_b" / f"kaggle_{tag}"
k_dir.mkdir(parents=True, exist_ok=True)

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
(k_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

nb = {
    "cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [f"# Standard Baseline\n", f"Automated execution: `{cmd_str}`\n"]},
        {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [
            "import os, sys, shutil, subprocess, json, torch\n",
            "from pathlib import Path\n",
            f"code_src = Path('/kaggle/input/datasets/{account}/tod-program-b-b2-code-20260814')\n",
            "if not code_src.exists():\n",
            f"    candidates = list(Path('/kaggle/input').glob('*{account}*')) + list(Path('/kaggle/input').glob('*tod-program-b-b2-code*'))\n",
            "    if candidates: code_src = candidates[0]\n",
            f"work_dir = Path('/kaggle/working/{tag}')\n",
            "if work_dir.exists(): shutil.rmtree(work_dir)\n",
            "shutil.copytree(code_src, work_dir)\n",
            "os.chdir(work_dir)\n",
            "sys.path.insert(0, str(work_dir))\n",
            "torch_cache = work_dir / 'torch_cache'\n",
            "if torch_cache.exists():\n",
            "    dst_cache = Path.home() / '.cache/torch'\n",
            "    dst_cache.mkdir(parents=True, exist_ok=True)\n",
            "    for f in (torch_cache / 'hub/checkpoints').glob('*.pth'):\n",
            "        (dst_cache / 'hub/checkpoints').mkdir(parents=True, exist_ok=True)\n",
            "        shutil.copy(f, dst_cache / 'hub/checkpoints' / f.name)\n",
            "    os.environ['TORCH_HOME'] = str(dst_cache)\n",
            "subprocess.run(['pip', 'install', '-q', 'aitodpycocotools', 'torchmetrics', 'pycocotools'], capture_output=True)\n",
            f"print('Executing command: {cmd_str}')\n",
            f"proc = subprocess.run({repr(cmd_str.split())}, stdout=sys.stdout, stderr=sys.stderr)\n",
            "print('Execution finished with exit code:', proc.returncode)\n",
            "assert proc.returncode == 0\n"
        ]}
    ],
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4, "nbformat_minor": 2
}
(k_dir / f"{tag}.ipynb").write_text(json.dumps(nb, indent=2) + "\n", encoding="utf-8")

cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(k_dir)]
res = subprocess.run(cmd, env=env, capture_output=True, text=True)
print(f"Push {account}:", res.stdout.strip())
if res.stderr.strip(): print(f"  Stderr {account}:", res.stderr.strip())
