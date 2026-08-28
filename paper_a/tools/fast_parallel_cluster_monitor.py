"""
High-Speed Parallel Multi-Account Cluster Monitor for All Kaggle Accounts.
Uses ThreadPoolExecutor to check all accounts concurrently in parallel.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"

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


def check_single_account(username: str, cred_path: Path) -> dict:
    if not cred_path.exists():
        return {"username": username, "status": "NO_CREDENTIALS", "kernels": []}
    try:
        creds = json.loads(cred_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {"username": username, "status": f"CRED_ERROR: {e}", "kernels": []}

    worker_code = f"""
import os, sys, json
from kaggle.api.kaggle_api_extended import KaggleApi

os.environ['KAGGLE_USERNAME'] = {repr(creds['username'])}
os.environ['KAGGLE_KEY'] = {repr(creds['key'])}

try:
    api = KaggleApi()
    api.authenticate()
    kernels = api.kernels_list(mine=True, sort_by='dateRun', page_size=6)
    records = []
    for k in kernels:
        ref = str(k.ref)
        try:
            st_obj = api.kernels_status(ref)
            st = st_obj.get('status', 'unknown') if isinstance(st_obj, dict) else str(st_obj)
        except Exception:
            st = 'unknown'
        
        latest_line = ""
        try:
            log_res = api.kernels_logs(ref)
            log_text = log_res.get('log', '') if isinstance(log_res, dict) else str(log_res)
            if log_text:
                lines = [l.strip() for l in log_text.splitlines() if l.strip()]
                metric_lines = [l for l in lines if ('Epoch ' in l and ('AP=' in l or 'mAP@50' in l or 'AP50=' in l or 'Loss=' in l or 'iter' in l.lower()))]
                if metric_lines:
                    latest_line = metric_lines[-1]
                elif lines:
                    latest_line = lines[-1][:120]
        except Exception:
            pass

        records.append({{
            'ref': ref,
            'title': getattr(k, 'title', ref.split('/')[-1]),
            'status': st,
            'latest': latest_line,
            'lastRunTime': str(getattr(k, 'lastRunTime', ''))
        }})
    print(json.dumps({{'username': '{username}', 'status': 'OK', 'kernels': records}}))
except Exception as e:
    print(json.dumps({{'username': '{username}', 'status': 'ERROR', 'error': str(e)}}))
"""
    res = subprocess.run([sys.executable, "-c", worker_code], capture_output=True, text=True, timeout=35)
    try:
        return json.loads(res.stdout.strip())
    except Exception:
        return {"username": username, "status": "PARSE_ERROR", "raw": res.stdout, "err": res.stderr}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 110)
    print("⚡ PARALLEL CLUSTER MONITOR: CHECKING ALL 12 KAGGLE GPU ACCOUNTS CONCURRENTLY...")
    print("=" * 110)

    results = {}
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_user = {
            executor.submit(check_single_account, user, path): user 
            for user, path in ACCOUNTS
        }
        for future in as_completed(future_to_user):
            user = future_to_user[future]
            try:
                data = future.result()
                results[user] = data
            except Exception as e:
                results[user] = {"username": user, "status": "EXCEPTION", "error": str(e)}

    # Summary
    running_count = 0
    queued_count = 0
    complete_count = 0
    error_count = 0

    print(f"\n{'ACCOUNT':<16} | {'LATEST KERNEL':<40} | {'STATUS':<12} | {'LAST METRIC / LOG'}")
    print("-" * 110)

    for username, _ in ACCOUNTS:
        info = results.get(username, {})
        status = info.get("status", "UNKNOWN")
        if status != "OK":
            print(f"{username:<16} | {'[AUTH / API ERROR]':<40} | {status:<12} | {info.get('error', info.get('err', ''))[:40]}")
            continue

        kernels = info.get("kernels", [])
        if not kernels:
            print(f"{username:<16} | {'(No kernels found)':<40} | {'IDLE':<12} | -")
            continue

        # Look for running/queued kernel first
        active = [k for k in kernels if any(s in k.get("status", "").lower() for s in ["running", "queued"])]
        if active:
            for k in active:
                st = k.get("status", "").upper()
                if "RUNNING" in st:
                    running_count += 1
                elif "QUEUED" in st:
                    queued_count += 1
                kernel_slug = k.get("ref", "").split("/")[-1]
                latest = k.get("latest", "-")
                print(f"🟢 {username:<14} | {kernel_slug:<40} | {st:<12} | {latest}")
        else:
            top = kernels[0]
            st = top.get("status", "").upper()
            if "COMPLETE" in st:
                complete_count += 1
            elif "ERROR" in st:
                error_count += 1
            kernel_slug = top.get("ref", "").split("/")[-1]
            latest = top.get("latest", "-")
            icon = "✅" if "COMPLETE" in st else ("❌" if "ERROR" in st else "⚪")
            print(f"{icon} {username:<14} | {kernel_slug:<40} | {st:<12} | {latest}")

    print("=" * 110)
    print(f"📊 SUMMARY: {running_count} Running | {queued_count} Queued | {complete_count} Complete (Latest) | {error_count} Error (Latest)")
    print("=" * 110)


if __name__ == "__main__":
    main()
