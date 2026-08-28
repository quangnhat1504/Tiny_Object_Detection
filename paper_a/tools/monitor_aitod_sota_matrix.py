"""
Multi-Account Real-Time Monitor and Artifact Downloader for Complete 12-GPU AI-TOD-v2 Matrix.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"
OUT_DIR = ROOT / "journal/results/aitod_empirical"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AITOD_12_RUNS = [
    {
        "id": "aitod_baseline_s42",
        "account": "amongus1504",
        "cred": "kaggle.json",
        "ref": "amongus1504/tod-aitod-baseline-s42-20260823",
        "name": "Faster R-CNN Baseline (ICCV'15, Fair S42)",
    },
    {
        "id": "aitod_nwd_s42",
        "account": "dipphmngc",
        "cred": "kaggle (11).json",
        "ref": "dipphmngc/tod-aitod-nwd-s42-20260823",
        "name": "NWD (NeurIPS'21, Fair S42)",
    },
    {
        "id": "aitod_igwd_s42",
        "account": "hienquang06",
        "cred": "kaggle (5).json",
        "ref": "hienquang06/tod-aitod-igwd-s42-20260823",
        "name": "IGWD (IEEE TMM'22, Fair S42)",
    },
    {
        "id": "aitod_rfla_s42",
        "account": "hngngnguynvn",
        "cred": "kaggle (1).json",
        "ref": "hngngnguynvn/tod-aitod-rfla-s42-20260823",
        "name": "RFLA (ECCV'22, Fair S42)",
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
    {
        "id": "aitod_cascade_s42",
        "account": "hngtrngtn",
        "cred": "kaggle (7).json",
        "ref": "hngtrngtn/tod-aitod-cascade-s42-20260824",
        "name": "Cascade R-CNN (CVPR'18, S42)",
    },
    {
        "id": "aitod_dotd_s42",
        "account": "luongsythanh",
        "cred": "kaggle (8).json",
        "ref": "luongsythanh/tod-aitod-dotd-s42-20260824",
        "name": "DotD (ICCV'21, S42)",
    },
    {
        "id": "aitod_simd_s42",
        "account": "pptlyn11",
        "cred": "kaggle (9).json",
        "ref": "pptlyn11/tod-aitod-simd-s42-20260824",
        "name": "SimD (CVPR'23, S42)",
    },
    {
        "id": "aitod_safit_s42",
        "account": "trieuvo123",
        "cred": "kaggle (10).json",
        "ref": "trieuvo123/tod-aitod-safit-s42-20260824",
        "name": "SAFit (AAAI'24, S42)",
    },
    {
        "id": "aitod_hwiou_cascade_s42",
        "account": "phuc1806",
        "cred": "kaggle (12).json",
        "ref": "phuc1806/tod-aitod-hwiou-cascade-s42-20260824",
        "name": "H-WIoU + Cascade Hybrid (Ours, S42)",
    },
]


def poll_kernel(run_info: dict):
    account = run_info["account"]
    cred_file = run_info["cred"]
    ref = run_info["ref"]
    name = run_info["name"]

    creds_path = CREDS_DIR / cred_file
    if not creds_path.exists():
        print(f"[!] Credential {cred_file} not found for {account}")
        return "UNKNOWN"

    creds = json.loads(creds_path.read_text(encoding="utf-8"))
    env = os.environ.copy()
    env["KAGGLE_USERNAME"] = creds["username"]
    env["KAGGLE_KEY"] = creds["key"]

    cmd = ["kaggle", "kernels", "status", ref]
    res = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    status_str = res.stdout.strip() if res.stdout else res.stderr.strip()
    print(f"[{name}] ({account}) -> {status_str}")
    return status_str


def main():
    print("=== Checking Status of All 12 Concurrent AI-TOD-v2 Experiments ===\n")
    for r in AITOD_12_RUNS:
        poll_kernel(r)
        time.sleep(0.5)


if __name__ == "__main__":
    main()
