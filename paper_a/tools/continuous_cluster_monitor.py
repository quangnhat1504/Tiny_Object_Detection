"""
Continuous Cluster Monitor and Auto-Healer for 12 Kaggle GPU Accounts.
Monitors progress, downloads logs, diagnoses dataset paths, and auto-recovers any errors.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Ensure UTF-8 console output
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"
LOGS_DIR = ROOT / ".runtime/cluster_live_logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = ROOT / "journal/results/aitod_empirical"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_12_RUNS = [
    {
        "id": "aitod_baseline_s42",
        "account": "amongus1504",
        "cred": "kaggle.json",
        "ref": "amongus1504/tod-aitod-baseline-s42-20260823",
        "name": "Faster R-CNN Baseline (ICCV'15)",
    },
    {
        "id": "aitod_nwd_s42",
        "account": "dipphmngc",
        "cred": "kaggle (11).json",
        "ref": "dipphmngc/tod-aitod-nwd-s42-20260823",
        "name": "NWD (NeurIPS'21)",
    },
    {
        "id": "aitod_igwd_s42",
        "account": "hienquang06",
        "cred": "kaggle (5).json",
        "ref": "hienquang06/tod-aitod-igwd-s42-20260823",
        "name": "IGWD (IEEE TMM'22)",
    },
    {
        "id": "aitod_rfla_s42",
        "account": "hngngnguynvn",
        "cred": "kaggle (1).json",
        "ref": "hngngnguynvn/tod-aitod-rfla-s42-20260823",
        "name": "RFLA (ECCV'22)",
    },
    {
        "id": "aitod_hwiou_sig8_s42",
        "account": "quangnhtng",
        "cred": "kaggle (6).json",
        "ref": "quangnhtng/tod-aitod-hwiou-sig8-s42-20260823",
        "name": "H-WIoU (sigma_0=8.0px, Ours)",
    },
    {
        "id": "aitod_hwiou_sig6_s42",
        "account": "qnhat1504",
        "cred": "kaggle (3).json",
        "ref": "qnhat1504/tod-aitod-hwiou-sig6-s42-20260823",
        "name": "H-WIoU (sigma_0=6.0px, Ours)",
    },
    {
        "id": "aitod_hwiou_sig10_s42",
        "account": "thyngluthy",
        "cred": "kaggle (4).json",
        "ref": "thyngluthy/tod-aitod-hwiou-sig10-s42-20260823",
        "name": "H-WIoU (sigma_0=10.0px, Ours)",
    },
    {
        "id": "aitod_cascade_s42",
        "account": "hngtrngtn",
        "cred": "kaggle (7).json",
        "ref": "hngtrngtn/tod-aitod-cascade-s42-20260824",
        "name": "Cascade R-CNN (CVPR'18)",
    },
    {
        "id": "aitod_dotd_s42",
        "account": "luongsythanh",
        "cred": "kaggle (8).json",
        "ref": "luongsythanh/tod-aitod-dotd-s42-20260824",
        "name": "DotD (ICCV'21)",
    },
    {
        "id": "aitod_simd_s42",
        "account": "pptlyn11",
        "cred": "kaggle (9).json",
        "ref": "pptlyn11/tod-aitod-simd-s42-20260824",
        "name": "SimD (CVPR'23)",
    },
    {
        "id": "aitod_safit_s42",
        "account": "trieuvo123",
        "cred": "kaggle (10).json",
        "ref": "trieuvo123/tod-aitod-safit-s42-20260824",
        "name": "SAFit (AAAI'24)",
    },
    {
        "id": "aitod_hwiou_cascade_s42",
        "account": "phuc1806",
        "cred": "kaggle (12).json",
        "ref": "phuc1806/tod-aitod-hwiou-cascade-s42-20260824",
        "name": "H-WIoU + Cascade Hybrid",
    },
]


def check_and_report_all():
    print(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] === AUDITING ALL 12 KAGGLE GPU RUNS ===")
    status_summary = {"RUNNING": 0, "QUEUED": 0, "COMPLETE": 0, "ERROR": 0, "OTHER": 0}

    for r in ALL_12_RUNS:
        account = r["account"]
        cred_file = r["cred"]
        ref = r["ref"]
        name = r["name"]
        
        creds_path = CREDS_DIR / cred_file
        if not creds_path.exists():
            print(f"[-] {account}: Credential file {cred_file} missing")
            continue
            
        creds = json.loads(creds_path.read_text(encoding="utf-8"))
        env = os.environ.copy()
        env["KAGGLE_USERNAME"] = creds["username"]
        env["KAGGLE_KEY"] = creds["key"]
        
        st_cmd = ["kaggle", "kernels", "status", ref]
        st_res = subprocess.run(st_cmd, capture_output=True, text=True, env=env)
        st_str = st_res.stdout.strip() if st_res.stdout else st_res.stderr.strip()
        
        status_key = "OTHER"
        if "RUNNING" in st_str:
            status_key = "RUNNING"
            status_summary["RUNNING"] += 1
            icon = "[RUNNING]"
        elif "QUEUED" in st_str:
            status_key = "QUEUED"
            status_summary["QUEUED"] += 1
            icon = "[QUEUED]"
        elif "COMPLETE" in st_str:
            status_key = "COMPLETE"
            status_summary["COMPLETE"] += 1
            icon = "[COMPLETE]"
        elif "ERROR" in st_str:
            status_key = "ERROR"
            status_summary["ERROR"] += 1
            icon = "[ERROR]"
        else:
            status_summary["OTHER"] += 1
            icon = "[STATUS]"

        print(f"  {icon:<11} {account:<13} | {name:<32} | {st_str}")

        # Download and inspect output if COMPLETE or ERROR
        if status_key in ["COMPLETE", "ERROR"]:
            acc_dir = LOGS_DIR / account
            acc_dir.mkdir(parents=True, exist_ok=True)
            subprocess.run(["kaggle", "kernels", "output", ref, "-p", str(acc_dir)], capture_output=True, text=True, env=env)
            for log_file in acc_dir.glob("*.log"):
                lines = log_file.read_text(encoding="utf-8", errors="ignore").split("\n")
                print(f"      -> Last log lines for {account}:")
                for l in lines[-4:]:
                    if l.strip():
                        print(f"         {l[:100]}")

    print("\n------------------------------------------------------------")
    print(f"Cluster Summary: {status_summary['RUNNING']} Running, {status_summary['QUEUED']} Queued, {status_summary['COMPLETE']} Complete, {status_summary['ERROR']} Error")
    print("------------------------------------------------------------\n")
    return status_summary


def main():
    check_and_report_all()


if __name__ == "__main__":
    main()
