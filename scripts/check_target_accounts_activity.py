"""
Check active kernel status on target candidate accounts.
"""
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"
PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_active_check")
PROFILE_ROOT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("qnhat1504", "kaggle (3).json"),
    ("thyngluthy", "kaggle (4).json"),
    ("hienquang06", "kaggle (5).json"),
    ("amongus1504", "kaggle (2).json"),
    ("dipphmngc", "kaggle (11).json"),
    ("hngngnguynvn", "kaggle (1).json"),
    ("pptlyn11", "kaggle (9).json"),
    ("trieuvo123", "kaggle (10).json"),
]

for uname, cname in TARGETS:
    cpath = CREDS_DIR / cname
    prof = PROFILE_ROOT / uname
    prof.mkdir(parents=True, exist_ok=True)
    target_cred = prof / "kaggle.json"
    target_cred.write_bytes(cpath.read_bytes())

    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(prof)

    # List running kernels
    res = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "list", "--mine", "--page-size", "5"],
        env=env, capture_output=True, text=True
    )
    print(f"\n--- Account: {uname} ---")
    lines = [line for line in res.stdout.strip().split("\n") if line.strip()]
    for l in lines[:4]:
        print("  ", l)
