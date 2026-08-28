"""
Inspect status and download latest logs for AI-TOD-v2 Kaggle training workers.
"""
from __future__ import annotations
import os
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_aitod_profiles")
OUTPUT_ROOT = ROOT / ".runtime/kaggle_aitod_outputs"
LOGS_DIR = ROOT / ".runtime/kaggle_aitod_logs"
PYTHON_EXEC = r"C:\Users\ADMIN\_Project\tiny-object-detection\.venv-cuda\Scripts\python.exe"

KERNELS = [
    {
        "name": "H-WIoU (sigma=8px, seed=42) [CANONICAL]",
        "account": "luongsythanh",
        "slug": "luongsythanh/tod-aitod-hwiou-sig8-s42",
    },
    {
        "name": "H-WIoU (sigma=6px, seed=42) [ABLATION]",
        "account": "hngtrngtn",
        "slug": "hngtrngtn/tod-aitod-hwiou-sig6-s42",
    },
    {
        "name": "H-WIoU (sigma=10px, seed=42) [ABLATION]",
        "account": "qnhat1504",
        "slug": "qnhat1504/tod-aitod-hwiou-sig10-s42",
    },
    {
        "name": "H-WIoU (sigma=8px, seed=123) [SEED-123]",
        "account": "hienquang06",
        "slug": "hienquang06/tod-aitod-hwiou-sig8-s123",
    },
    {
        "name": "H-WIoU (sigma=8px, seed=2024) [SEED-2024]",
        "account": "quangnhtng",
        "slug": "quangnhtng/tod-aitod-hwiou-sig8-s2024",
    },
]

def main():
    print("=" * 80)
    print("   AI-TOD-v2 GPU CLUSTER STATUS & RECENT TRAINING LOGS")
    print("=" * 80)

    for k in KERNELS:
        name = k["name"]
        account = k["account"]
        slug = k["slug"]
        profile = PROFILE_ROOT / account

        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(profile)

        print(f"\n[+] Account: {account} | Kernel: {slug}")
        print(f"    Target: {name}")

        # Check status
        cmd_stat = [PYTHON_EXEC, "-m", "kaggle", "kernels", "status", slug]
        res_stat = subprocess.run(cmd_stat, env=env, capture_output=True, text=True, timeout=30)
        status_text = res_stat.stdout.strip() if res_stat.stdout else res_stat.stderr.strip()
        print(f"    Status: {status_text}")

        # Try downloading output
        dest_dir = OUTPUT_ROOT / f"{account}_{slug.split('/')[-1]}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        cmd_out = [PYTHON_EXEC, "-m", "kaggle", "kernels", "output", slug, "-p", str(dest_dir)]
        res_out = subprocess.run(cmd_out, env=env, capture_output=True, text=True, timeout=60)
        
        files = list(dest_dir.glob("*"))
        print(f"    Output files in {dest_dir.name} ({len(files)}): {[f.name for f in files]}")

        # Read any text logs
        for lf in dest_dir.glob("*.log"):
            print(f"    --- Last lines of {lf.name} ---")
            lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()
            for l in lines[-10:]:
                print(f"      {l}")

        for jf in dest_dir.glob("*.json"):
            if jf.name != "kernel-metadata.json":
                print(f"    --- Content of {jf.name} ---")
                print(f"      {jf.read_text(encoding='utf-8', errors='ignore')[:300]}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
