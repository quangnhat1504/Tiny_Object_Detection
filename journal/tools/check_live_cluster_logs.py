"""
Fetch live console logs and epoch progress for all active Kaggle AI-TOD training kernels.
"""
import os
import json
import subprocess
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDENTIALS_DIR = ROOT / ".runtime/kaggle_credentials"

WORKERS = [
    ("luongsythanh", "tod-aitod-hwiou-sig8-s42", "H-WIoU (sigma=8px, seed=42) [CANONICAL]"),
    ("hngtrngtn", "tod-aitod-hwiou-sig6-s42", "H-WIoU (sigma=6px, seed=42) [ABLATION]"),
    ("qnhat1504", "tod-aitod-hwiou-sig10-s42", "H-WIoU (sigma=10px, seed=42) [ABLATION]"),
    ("hienquang06", "tod-aitod-hwiou-sig8-s123", "H-WIoU (sigma=8px, seed=123) [MULTI-SEED]"),
    ("quangnhtng", "tod-aitod-hwiou-sig8-s2024", "H-WIoU (sigma=8px, seed=2024) [MULTI-SEED]")
]

def main():
    print("=" * 80)
    print("   LIVE EPOCH PROGRESS & CONSOLE LOGS ACROSS KAGGLE T4 CLUSTER")
    print("=" * 80)

    for account, slug, desc in WORKERS:
        cred_file = CREDENTIALS_DIR / f"{account}.json"
        if not cred_file.exists():
            print(f"[-] Missing credentials for {account}")
            continue

        with open(cred_file, "r") as f:
            cred = json.load(f)

        env = os.environ.copy()
        env["KAGGLE_USERNAME"] = cred.get("username", account)
        env["KAGGLE_KEY"] = cred.get("key", "")

        print(f"\n>>> [{account}] {desc}: {slug}")
        
        # 1. Check status
        try:
            status_proc = subprocess.run(
                ["kaggle", "kernels", "status", f"{account}/{slug}"],
                env=env, capture_output=True, text=True, timeout=30
            )
            print(f"    Status: {status_proc.stdout.strip()}")
        except Exception as e:
            print(f"    Status error: {e}")

        # 2. Try fetching output/logs if any
        log_dir = ROOT / f".runtime/kaggle_logs/{account}_{slug}"
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            out_proc = subprocess.run(
                ["kaggle", "kernels", "output", f"{account}/{slug}", "-p", str(log_dir)],
                env=env, capture_output=True, text=True, timeout=60
            )
            files = list(log_dir.glob("*"))
            print(f"    Downloaded files ({len(files)}): {[f.name for f in files]}")
            
            # Check log text files
            log_files = list(log_dir.glob("*.log")) + list(log_dir.glob("*.txt")) + list(log_dir.glob("*.csv")) + list(log_dir.glob("*.json"))
            for lf in log_files:
                print(f"    --- Last lines of {lf.name} ---")
                lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()
                for l in lines[-8:]:
                    print(f"      {l}")
        except Exception as e:
            print(f"    Output fetch error: {e}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
