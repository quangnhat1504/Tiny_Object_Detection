"""
Automated downloader for Official AI-TOD Dataset to D:\paper_a_data\AI-TOD-v2
"""
import os
import subprocess
import sys
import time
from pathlib import Path

TARGET_DIR = Path(r"D:\paper_a_data\AI-TOD-v2")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

CREDS = Path.home() / ".kaggle" / "kaggle (2).json"
PROFILE = Path(r"C:\tmp\tod_kaggle_aitod_download")
PROFILE.mkdir(parents=True, exist_ok=True)

import shutil
shutil.copy(CREDS, PROFILE / "kaggle.json")

env = os.environ.copy()
env["KAGGLE_CONFIG_DIR"] = str(PROFILE)

print("=" * 80)
print(f"=== DOWNLOADING OFFICIAL AI-TOD DATASET TO {TARGET_DIR} ===")
print("Source: simplestzyp/tiny-object-detection-in-aerial-images (25.2 GB)")
print("=" * 80)

cmd = [
    sys.executable, "-m", "kaggle", "datasets", "download",
    "-d", "simplestzyp/tiny-object-detection-in-aerial-images",
    "-p", str(TARGET_DIR),
    "--unzip"
]

start_t = time.time()
res = subprocess.run(cmd, env=env)
elapsed = time.time() - start_t

if res.returncode == 0:
    print(f"\n[SUCCESS] AI-TOD dataset downloaded and extracted in {elapsed/60:.2f} minutes!")
else:
    print(f"\n[ERROR] Download failed with return code {res.returncode}")
