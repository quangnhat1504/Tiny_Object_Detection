"""
Download all completed Journal and AI-TOD benchmark artifacts and parse full multi-metric tables.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"
RESULTS_DIR = ROOT / "journal/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNTS = [
    ("amongus1504", CREDS_DIR / "kaggle.json", "tod-aitod-baseline-s42-20260820", "AI-TOD-v2 Baseline (12 Ep)"),
    ("quangnhtng", CREDS_DIR / "kaggle (6).json", "tod-aitod-hwiou-sig8-s42-20260820", "AI-TOD-v2 H-WIoU sig8 (12 Ep)"),
    ("qnhat1504", CREDS_DIR / "kaggle (3).json", "tod-aitod-hwiou-sig6-s42-20260820", "AI-TOD-v2 H-WIoU sig6 (12 Ep)"),
    ("thyngluthy", CREDS_DIR / "kaggle (4).json", "tod-tp-ablation-pure-w2-s42-20260820", "TinyPerson Ablation Pure W2"),
    ("hngngnguynvn", CREDS_DIR / "kaggle (1).json", "tod-tp-ablation-pure-iou-s42-20260820", "TinyPerson Ablation Pure IoU"),
    ("dipphmngc", CREDS_DIR / "kaggle (11).json", "tod-tp-ablation-static-half-s42-20260820", "TinyPerson Ablation Static 0.5"),
]


def download_run(username: str, cred_path: Path, slug: str, name: str):
    creds = json.loads(cred_path.read_text(encoding="utf-8"))
    out_dir = RESULTS_DIR / f"{username}_{slug}"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = creds["username"]
    env["KAGGLE_KEY"] = creds["key"]

    ref = f"{username}/{slug}"
    print(f"\nDownloading outputs for [{name}] ({ref}) ...")
    cmd = [sys.executable, "-m", "kaggle", "kernels", "output", ref, "-p", str(out_dir)]
    subprocess.run(cmd, env=env, capture_output=True)

    # Inspect downloaded files
    metrics_files = list(out_dir.rglob("*.json")) + list(out_dir.rglob("*.log"))
    print(f"  Found {len(list(out_dir.iterdir()))} files in {out_dir.name}")
    for mf in metrics_files:
        print(f"    - {mf.name} ({mf.stat().st_size} bytes)")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 105)
    print("                DOWNLOADING & EXTRACTING ALL JOURNAL & AI-TOD-v2 CLUSTER RUNS                ")
    print("=" * 105)

    for username, cred_path, slug, name in ACCOUNTS:
        download_run(username, cred_path, slug, name)

    print("\n" + "=" * 105)


if __name__ == "__main__":
    main()
