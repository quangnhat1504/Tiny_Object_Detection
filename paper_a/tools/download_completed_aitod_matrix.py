"""
Download completed AI-TOD / Journal SOTA benchmark runs and parse all test metrics.
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

COMPLETED_RUNS = [
    ("hngtrngtn", CREDS_DIR / "kaggle (7).json", "tod-aitod-cascade-s42-20260824", "Cascade R-CNN Baseline"),
    ("amongus1504", CREDS_DIR / "kaggle.json", "tod-aitod-baseline-s42-20260823", "Faster R-CNN Baseline"),
    ("hngngnguynvn", CREDS_DIR / "kaggle (1).json", "tod-aitod-rfla-s42-20260823", "RFLA (Gaussian Assignment)"),
    ("dipphmngc", CREDS_DIR / "kaggle (11).json", "tod-aitod-nwd-s42-20260823", "NWD (Wasserstein Distance)"),
    ("hienquang06", CREDS_DIR / "kaggle (5).json", "tod-aitod-igwd-s42-20260823", "IGWD (Gaussian Wasserstein)"),
]


def download_run_artifacts(username: str, cred_path: Path, slug: str, label: str):
    if not cred_path.exists():
        print(f"❌ Missing credential file for {username}: {cred_path}")
        return None
    creds = json.loads(cred_path.read_text(encoding="utf-8"))
    ref = f"{username}/{slug}"
    target_dir = RESULTS_DIR / f"{username}_{slug}"
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"📥 Downloading: {label}")
    print(f"   Kernel Reference: {ref}")
    print(f"   Destination: {target_dir}")
    print(f"=======================================================")

    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = creds["username"]
    env["KAGGLE_KEY"] = creds["key"]

    # 1. Download kernel output files
    cmd = [sys.executable, "-m", "kaggle", "kernels", "output", ref, "-p", str(target_dir)]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"   [Output Warning]: {res.stderr.strip()}")
    else:
        print(f"   [Output Download]: Success")

    # 2. Also fetch full log file
    log_cmd = [
        sys.executable, "-c",
        f"""
import os, sys, json
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
try:
    res = api.kernels_logs('{ref}')
    log = res.get('log', '') if isinstance(res, dict) else str(res)
    print(log)
except Exception as e:
    print('ERROR:', e)
"""
    ]
    log_res = subprocess.run(log_cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if log_res.stdout.strip():
        log_file = target_dir / f"{slug}.log"
        log_file.write_text(log_res.stdout, encoding="utf-8")
        print(f"   [Log Download]: Saved {len(log_res.stdout)} chars to {log_file.name}")

    # Inspect downloaded files
    files = list(target_dir.rglob("*"))
    print(f"   Artifacts in {target_dir.name}: {len(files)} items")
    for f in sorted(files):
        if f.is_file():
            print(f"     * {f.relative_to(target_dir)} ({f.stat().st_size:,} bytes)")

    return target_dir


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for username, cred_path, slug, label in COMPLETED_RUNS:
        download_run_artifacts(username, cred_path, slug, label)


if __name__ == "__main__":
    main()
