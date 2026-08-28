"""
High-Speed Concurrent Downloader for all completed AI-TOD Benchmark Runs.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

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


def download_single(username: str, cred_path: Path, slug: str, label: str):
    if not cred_path.exists():
        return {"user": username, "status": "NO_CREDS"}
    creds = json.loads(cred_path.read_text(encoding="utf-8"))
    ref = f"{username}/{slug}"
    target_dir = RESULTS_DIR / f"{username}_{slug}"
    target_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = creds["username"]
    env["KAGGLE_KEY"] = creds["key"]

    # 1. Download output
    cmd = [sys.executable, "-m", "kaggle", "kernels", "output", ref, "-p", str(target_dir)]
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)

    # 2. Download logs
    log_cmd = [
        sys.executable, "-c",
        f"""
import os, sys
from kaggle.api.kaggle_api_extended import KaggleApi
os.environ['KAGGLE_USERNAME'] = {repr(creds['username'])}
os.environ['KAGGLE_KEY'] = {repr(creds['key'])}
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
        (target_dir / f"{slug}.log").write_text(log_res.stdout, encoding="utf-8")

    files = list(target_dir.rglob("*"))
    file_count = len([f for f in files if f.is_file()])
    return {
        "user": username,
        "label": label,
        "ref": ref,
        "dir": str(target_dir.name),
        "files": file_count,
        "status": "OK" if res.returncode == 0 else "WARN"
    }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 90)
    print("⚡ PARALLEL DOWNLOADER: DOWNLOADING ALL COMPLETED KERNELS CONCURRENTLY...")
    print("=" * 90)

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [
            executor.submit(download_single, u, cp, s, l)
            for u, cp, s, l in COMPLETED_RUNS
        ]
        for f in as_completed(futures):
            res = f.result()
            print(f"✅ Downloaded [{res['label']}] ({res['ref']}) -> {res['files']} files in {res['dir']}")

    print("=" * 90)
    print("ALL DOWNLOADS COMPLETE!")


if __name__ == "__main__":
    main()
