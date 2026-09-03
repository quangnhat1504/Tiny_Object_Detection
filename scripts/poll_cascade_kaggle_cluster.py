"""
Poll status of all active Cascade, RFLA, DU-HWIoU, and QFL experiments on Kaggle GPU pool.
"""
import os
import subprocess
from pathlib import Path

CASCADE_KERNELS = [
    ("Cascade Homotopy Proposed", "luongsythanh", "tod-cascade-homotopy-s42-proposed", Path(r"C:\tmp\tod_kaggle_cascade_profiles")),
    ("Cascade Baseline Standard", "amongus1504", "tod-cascade-baseline-s42", Path(r"C:\tmp\tod_kaggle_cascade_profiles")),
    ("RFLA + H-WIoU Proposed", "qnhat1504", "tod-rfla-hwiou-s42-proposed", Path(r"C:\tmp\tod_kaggle_cascade_profiles")),
    ("RFLA + Smooth-L1 Baseline", "thyngluthy", "tod-rfla-baseline-s42", Path(r"C:\tmp\tod_kaggle_cascade_profiles")),
    ("DU-HWIoU Proposed", "quangnhtng", "tod-aitod-du-hwiou-s42-proposed", Path(r"C:\tmp\tod_kaggle_wave2_profiles")),
    ("QFL + H-WIoU Proposed", "trieuvo123", "tod-aitod-qfl-hwiou-s42-proposed", Path(r"C:\tmp\tod_kaggle_wave2_profiles")),
]

def check_status():
    print("=" * 80)
    print("=== STATUS OF ALL 6 TOD BENCHMARK EXPERIMENTS ON KAGGLE GPU POOL ===")
    print("=" * 80)
    
    all_done = True
    for name, account, slug, profile_root in CASCADE_KERNELS:
        profile = profile_root / account
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(profile)
        
        full_slug = f"{account}/{slug}"
        cmd = ["kaggle", "kernels", "status", full_slug]
        res = subprocess.run(cmd, env=env, capture_output=True, text=True)
        status = res.stdout.strip() if res.returncode == 0 else res.stderr.strip()
        print(f"[{account:<14}] {name:<30}: {status}")
        if "complete" not in status.lower():
            all_done = False

    return all_done

if __name__ == "__main__":
    check_status()
