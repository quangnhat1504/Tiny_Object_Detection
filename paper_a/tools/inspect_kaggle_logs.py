"""Inspect latest output logs from active Kaggle kernels."""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"

AITOD_RUNS = [
    {
        "id": "aitod_baseline_s42",
        "account": "amongus1504",
        "cred": "kaggle.json",
        "ref": "amongus1504/tod-aitod-baseline-s42-20260823",
        "name": "Faster R-CNN Baseline (AI-TOD-v2, Fair S42)",
    },
    {
        "id": "aitod_nwd_s42",
        "account": "dipphmngc",
        "cred": "kaggle (11).json",
        "ref": "dipphmngc/tod-aitod-nwd-s42-20260823",
        "name": "NWD (NeurIPS 2021, Fair S42)",
    },
    {
        "id": "aitod_igwd_s42",
        "account": "hienquang06",
        "cred": "kaggle (5).json",
        "ref": "hienquang06/tod-aitod-igwd-s42-20260823",
        "name": "IGWD (IEEE TMM 2022, Fair S42)",
    },
    {
        "id": "aitod_rfla_s42",
        "account": "hngngnguynvn",
        "cred": "kaggle (1).json",
        "ref": "hngngnguynvn/tod-aitod-rfla-s42-20260823",
        "name": "RFLA (ECCV 2022, Fair S42)",
    },
    {
        "id": "aitod_hwiou_sig8_s42",
        "account": "quangnhtng",
        "cred": "kaggle (6).json",
        "ref": "quangnhtng/tod-aitod-hwiou-sig8-s42-20260823",
        "name": "H-WIoU Proposed (sigma_0=8.0px, S42)",
    },
    {
        "id": "aitod_hwiou_sig6_s42",
        "account": "qnhat1504",
        "cred": "kaggle (3).json",
        "ref": "qnhat1504/tod-aitod-hwiou-sig6-s42-20260823",
        "name": "H-WIoU Proposed (sigma_0=6.0px, S42)",
    },
    {
        "id": "aitod_hwiou_sig10_s42",
        "account": "thyngluthy",
        "cred": "kaggle (4).json",
        "ref": "thyngluthy/tod-aitod-hwiou-sig10-s42-20260823",
        "name": "H-WIoU Proposed (sigma_0=10.0px, S42)",
    },
]

for run in AITOD_RUNS:
    cred_file = CREDS_DIR / run["cred"]
    if not cred_file.exists():
        continue
    with open(cred_file, "r") as f:
        creds = json.load(f)
    
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = creds["username"]
    env["KAGGLE_KEY"] = creds["key"]
    
    print(f"\n=======================================================")
    print(f"[{run['name']}] ({run['account']}) -> {run['ref']}")
    print(f"=======================================================")
    
    # 1. Status
    res_st = subprocess.run(
        ["kaggle", "kernels", "status", run["ref"]],
        capture_output=True,
        text=True,
        env=env,
    )
    print(f"Status: {res_st.stdout.strip() if res_st.stdout else res_st.stderr.strip()}")
    
    # 2. Output Tail
    res = subprocess.run(
        ["kaggle", "kernels", "output", run["ref"]],
        capture_output=True,
        text=True,
        env=env,
    )
    lines = [l for l in (res.stdout or "").strip().split("\n") if l.strip()]
    if lines:
        print("Last log lines:")
        for l in lines[-6:]:
            print(f"  {l}")
    else:
        print("  (Log output buffering or kernel initializing)")
