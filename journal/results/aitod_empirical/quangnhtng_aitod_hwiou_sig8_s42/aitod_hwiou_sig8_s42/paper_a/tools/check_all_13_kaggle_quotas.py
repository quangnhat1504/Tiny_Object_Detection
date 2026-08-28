"""
Comprehensive Quota, Health, and Active Kernel Monitor for all 13 Kaggle Accounts.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"

# 13 Unique Accounts Mapping
ACCOUNTS = {
    "amongus1504": CREDS_DIR / "kaggle.json",
    "dipphmngc": CREDS_DIR / "kaggle (11).json",
    "hienquang06": CREDS_DIR / "kaggle (5).json",
    "hngngnguynvn": CREDS_DIR / "kaggle (1).json",
    "hngtrngtn": CREDS_DIR / "kaggle (7).json",
    "luongsythanh": CREDS_DIR / "kaggle (8).json",
    "ngquangnht": ROOT / ".runtime/kaggle/wp02/multi_account/cfg_ngquangnht/kaggle.json",
    "phuc1806": CREDS_DIR / "kaggle (12).json",
    "pptlyn11": CREDS_DIR / "kaggle (9).json",
    "qnhat1504": CREDS_DIR / "kaggle (3).json",
    "quangnhtng": CREDS_DIR / "kaggle (6).json",
    "thyngluthy": CREDS_DIR / "kaggle (4).json",
    "trieuvo123": CREDS_DIR / "kaggle (10).json",
}


def probe_account(username: str, cred_path: Path) -> dict:
    if not cred_path.exists():
        return {
            "username": username,
            "status": "MISSING_CRED_FILE",
            "active_kernels": [],
            "recent_kernels": [],
        }

    # Setup isolated temp profile
    with tempfile.TemporaryDirectory() as tmp_prof:
        tmp_cfg = Path(tmp_prof) / "kaggle.json"
        tmp_cfg.write_text(cred_path.read_text(encoding="utf-8"), encoding="utf-8")

        script = f"""
import os, sys, json
from kaggle.api.kaggle_api_extended import KaggleApi

os.environ['KAGGLE_CONFIG_DIR'] = {repr(tmp_prof)}
try:
    api = KaggleApi()
    api.authenticate()
except Exception as e:
    print(json.dumps({{'status': 'AUTH_FAILED', 'error': str(e)}}))
    sys.exit(0)

try:
    kernels = api.kernels_list(user='{username}', sort_by='dateRun', page_size=10)
    kernel_list = []
    active = []
    for k in kernels:
        ref = str(k.ref)
        try:
            st = api.kernels_status(ref)
            status_str = st.get('status', 'unknown') if isinstance(st, dict) else str(st)
        except Exception:
            status_str = 'unknown'
        
        entry = {{
            'ref': ref,
            'title': str(k.title),
            'status': status_str,
            'last_run': str(k.lastRunTime) if hasattr(k, 'lastRunTime') else '',
        }}
        kernel_list.append(entry)
        if any(x in status_str.lower() for x in ['running', 'queued']):
            active.append(entry)

    print(json.dumps({{
        'status': 'HEALTHY',
        'active_kernels': active,
        'recent_kernels': kernel_list,
    }}))
except Exception as e:
    print(json.dumps({{'status': 'API_ERROR', 'error': str(e)}}))
"""

        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(tmp_prof)
        res = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
        out = res.stdout.strip()
        try:
            return {"username": username, **json.loads(out)}
        except Exception:
            return {
                "username": username,
                "status": "PROBE_FAILED",
                "raw_stdout": out,
                "raw_stderr": res.stderr.strip(),
            }


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 110)
    print("                      COMPREHENSIVE 13-ACCOUNT KAGGLE STATUS & QUOTA REPORT                      ")
    print("=" * 110)

    results = []
    for idx, (user, p) in enumerate(ACCOUNTS.items(), 1):
        print(f"[{idx:02d}/13] Checking {user:<15} ...", end=" ", flush=True)
        res = probe_account(user, p)
        results.append(res)
        st = res.get("status", "UNKNOWN")
        active_count = len(res.get("active_kernels", []))
        print(f"Status: {st} | Active Running Kernels: {active_count}")

    print("\n" + "=" * 110)
    print(f"{'No.':<4} | {'Account Username':<16} | {'Auth Status':<12} | {'Active GPU Tasks':<18} | {'Recent Kernel & Status':<45}")
    print("-" * 110)

    for i, r in enumerate(results, 1):
        u = r["username"]
        st = r.get("status", "ERROR")
        actives = r.get("active_kernels", [])
        act_str = f"{len(actives)} RUNNING" if actives else "0 (Idle / Free)"
        recents = r.get("recent_kernels", [])
        if recents:
            top = recents[0]
            top_status = top.get("status", "")
            if "running" in top_status.lower():
                top_st_clean = "RUNNING"
            elif "complete" in top_status.lower():
                top_st_clean = "COMPLETE"
            elif "error" in top_status.lower() or "fail" in top_status.lower():
                top_st_clean = "ERROR"
            else:
                top_st_clean = top_status[:12]
            recent_str = f"{top['ref'].split('/')[-1][:28]} [{top_st_clean}]"
        else:
            recent_str = "No recent kernels"

        print(f"{i:<4} | {u:<16} | {st:<12} | {act_str:<18} | {recent_str:<45}")

    print("=" * 110)

    # Summary
    total_active = sum(len(r.get("active_kernels", [])) for r in results)
    total_healthy = sum(1 for r in results if r.get("status") == "HEALTHY")
    idle_accounts = [r["username"] for r in results if r.get("status") == "HEALTHY" and not r.get("active_kernels")]

    print(f"\n[SUMMARY ANALYSIS]")
    print(f"  * Total Kaggle Accounts Configured : 13 / 13")
    print(f"  * Total Healthy & Authenticated    : {total_healthy} / 13")
    print(f"  * Total Active GPU Kernels Running : {total_active}")
    print(f"  * Total Available Idle Accounts    : {len(idle_accounts)} / 13 ({', '.join(idle_accounts)})")
    print("=" * 110)


if __name__ == "__main__":
    main()
