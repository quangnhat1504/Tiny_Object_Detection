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
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_hwiou_profiles")

# Ensure latest code in stage
shutil.copy(ROOT / "common/metrics/h_wiou.py", STAGE_DIR / "common/metrics/h_wiou.py")
shutil.copy(ROOT / "common/model.py", STAGE_DIR / "common/model.py")
shutil.copy(ROOT / "scripts/train_frcnn_aitod.py", STAGE_DIR / "scripts/train_frcnn_aitod.py")

ACCOUNTS = [
    ("qnhat1504", "kaggle (3).json", "aitod_hwiou_sig6_s42", "tod-aitod-hwiou-sig6-s42-20260820"),
    ("amongus1504", "kaggle (2).json", "aitod_baseline_s42", "tod-aitod-baseline-s42-20260820"),
    ("thyngluthy", "kaggle (4).json", "tp_ablation_pure_w2_s42", "tod-tp-ablation-pure-w2-s42-20260820"),
    ("hngngnguynvn", "kaggle (1).json", "tp_ablation_pure_iou_s42", "tod-tp-ablation-pure-iou-s42-20260820"),
    ("dipphmngc", "kaggle (11).json", "tp_ablation_static_half_s42", "tod-tp-ablation-static-half-s42-20260820"),
]

for account, cred_file, tag, slug in ACCOUNTS:
    profile = PROFILE_ROOT / account
    profile.mkdir(parents=True, exist_ok=True)
    shutil.copy(CREDS_DIR / cred_file, profile / "kaggle.json")
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(profile)

    print(f"\nUpdating code dataset for {account}...")
    cmd = [sys.executable, "-m", "kaggle", "datasets", "version", "-p", str(STAGE_DIR), "--dir-mode", "zip", "-m", "Sync loss signature fix"]
    subprocess.run(cmd, env=env, capture_output=True)

    time.sleep(3)
    kernel_dir = ROOT / ".runtime/kaggle" / tag
    if kernel_dir.exists():
        print(f"Re-pushing kernel {account}/{slug}...")
        push_cmd = [sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kernel_dir)]
        res = subprocess.run(push_cmd, env=env, capture_output=True, text=True)
        print("Push Output:", res.stdout.strip())
        time.sleep(2)
        stat_cmd = [sys.executable, "-m", "kaggle", "kernels", "status", f"{account}/{slug}"]
        print("Status:", subprocess.run(stat_cmd, env=env, capture_output=True, text=True).stdout.strip())

print("\nAll accounts synced and kernels launched successfully!")
