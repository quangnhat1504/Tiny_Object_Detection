"""
Direct Multi-Account Real-Time Kernel Monitor & Metric Extractor for All 13 Kaggle Accounts.
Sets KAGGLE_USERNAME and KAGGLE_KEY per account dynamically.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"
RESULTS_DIR = ROOT / "journal/results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ACCOUNTS = [
    ("amongus1504", CREDS_DIR / "kaggle.json"),
    ("quangnhtng", CREDS_DIR / "kaggle (6).json"),
    ("qnhat1504", CREDS_DIR / "kaggle (3).json"),
    ("thyngluthy", CREDS_DIR / "kaggle (4).json"),
    ("hngngnguynvn", CREDS_DIR / "kaggle (1).json"),
    ("dipphmngc", CREDS_DIR / "kaggle (11).json"),
    ("hienquang06", CREDS_DIR / "kaggle (5).json"),
    ("trieuvo123", CREDS_DIR / "kaggle (10).json"),
    ("phuc1806", CREDS_DIR / "kaggle (12).json"),
    ("pptlyn11", CREDS_DIR / "kaggle (9).json"),
    ("luongsythanh", CREDS_DIR / "kaggle (8).json"),
    ("hngtrngtn", CREDS_DIR / "kaggle (7).json"),
]


def check_account_live(username: str, cred_path: Path):
    if not cred_path.exists():
        return
    creds = json.loads(cred_path.read_text(encoding="utf-8"))
    
    script = f"""
import os, sys, json
from kaggle.api.kaggle_api_extended import KaggleApi

os.environ['KAGGLE_USERNAME'] = {repr(creds['username'])}
os.environ['KAGGLE_KEY'] = {repr(creds['key'])}

try:
    api = KaggleApi()
    api.authenticate()
    kernels = api.kernels_list(mine=True, sort_by='dateRun', page_size=5)
    records = []
    for k in kernels:
        ref = str(k.ref)
        try:
            st_obj = api.kernels_status(ref)
            st = st_obj.get('status', 'unknown') if isinstance(st_obj, dict) else str(st_obj)
        except Exception:
            st = 'unknown'
        
        # Check logs if running or recently updated
        latest_line = ""
        try:
            log_res = api.kernels_logs(ref)
            log_text = log_res.get('log', '') if isinstance(log_res, dict) else str(log_res)
            if log_text:
                lines = [l.strip() for l in log_text.splitlines() if l.strip()]
                metric_lines = [l for l in lines if ('Epoch ' in l and ('AP=' in l or 'mAP@50' in l or 'AP50=' in l))]
                if metric_lines:
                    latest_line = metric_lines[-1]
                elif lines:
                    latest_line = lines[-1][:120]
        except Exception:
            pass

        records.append({{
            'ref': ref,
            'status': st,
            'latest': latest_line
        }})
    print(json.dumps({{'username': '{username}', 'status': 'OK', 'kernels': records}}))
except Exception as e:
    print(json.dumps({{'username': '{username}', 'status': 'ERROR', 'error': str(e)}}))
"""
    res = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    try:
        data = json.loads(res.stdout.strip())
        return data
    except Exception:
        return {"username": username, "status": "PARSE_ERROR", "raw": res.stdout}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 115)
    print("                 COMPREHENSIVE MULTI-ACCOUNT CLOUD GPU T4 LIVE STATUS & METRICS MONITOR                 ")
    print("=" * 115)

    for username, cred_path in ACCOUNTS:
        res = check_account_live(username, cred_path)
        if not res or res.get("status") != "OK":
            continue
        kernels = res.get("kernels", [])
        active = [k for k in kernels if any(x in k.get("status", "").lower() for x in ["running", "queued"])]
        if active:
            print(f"\n🟢 [{username.upper()}] -> {len(active)} Active Kernel(s) in Progress:")
            for k in active:
                print(f"   * Kernel : {k['ref']}")
                print(f"     Status : {k['status'].upper()}")
                if k.get("latest"):
                    print(f"     Metrics: {k['latest']}")
        else:
            top = kernels[0] if kernels else None
            if top and "tod" in top["ref"].lower():
                print(f"\n⚪ [{username:<12}] -> Latest: {top['ref']} [{top['status']}]")
                if top.get("latest"):
                    print(f"     Tail: {top['latest']}")

    print("\n" + "=" * 115)


if __name__ == "__main__":
    main()
