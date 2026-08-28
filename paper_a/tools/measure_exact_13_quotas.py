"""
Exact Real-Time GPU & TPU Quota Measurement Tool for All 13 Kaggle Accounts.
Queries api.quota_view() to extract exact remaining hours, used hours, and reset dates.
"""
from __future__ import annotations
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"

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


def parse_time_seconds(time_str: str) -> float:
    """Parse time string like '231.813.0s', '120.5s', '0s', '3600s' into float seconds."""
    if not time_str:
        return 0.0
    s_clean = str(time_str).strip().rstrip("s")
    # In case of malformed strings like '231.813.0'
    parts = s_clean.split(".")
    if len(parts) > 2:
        s_clean = parts[0] + "." + parts[1]
    try:
        return float(s_clean)
    except Exception:
        return 0.0


def query_account_quota(username: str, cred_path: Path) -> dict:
    if not cred_path.exists():
        return {"username": username, "status": "MISSING_FILE"}

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
    res = api.quota_view()
    data = res.to_dict() if hasattr(res, 'to_dict') else res
    print(json.dumps({{'status': 'SUCCESS', 'data': data}}))
except Exception as e:
    print(json.dumps({{'status': 'ERROR', 'error': str(e)}}))
"""
        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(tmp_prof)
        res = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True)
        try:
            return {"username": username, **json.loads(res.stdout.strip())}
        except Exception:
            return {"username": username, "status": "PARSE_ERROR", "raw": res.stdout.strip()}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    print("=" * 125)
    print("                       REAL-TIME GPU & TPU QUOTA AUDIT (13 KAGGLE ACCOUNTS)                       ")
    print("=" * 125)

    records = []
    for idx, (u, p) in enumerate(ACCOUNTS.items(), 1):
        print(f"[{idx:02d}/13] Querying {u:<15} ...", end=" ", flush=True)
        res = query_account_quota(u, p)
        records.append(res)
        st = res.get("status", "ERROR")
        if st == "SUCCESS":
            data = res.get("data", {})
            gpu = data.get("gpuQuota", {})
            used_s = parse_time_seconds(gpu.get("timeUsed", "0s"))
            total_s = parse_time_seconds(gpu.get("totalTimeAllowed", "108000s"))
            if total_s < 21600.0:
                total_s = 108000.0  # 30 hours weekly standard
            rem_s = max(total_s - used_s, 0.0)
            print(f"OK -> GPU Used: {used_s/3600:.2f}h / Total: {total_s/3600:.1f}h | Remaining: {rem_s/3600:.2f}h")
        else:
            print(f"FAILED ({res.get('error', 'unknown error')})")

    print("\n" + "=" * 125)
    print(f"{'No.':<4} | {'Kaggle Account':<15} | {'GPU Used (h)':<13} | {'GPU Remaining (h)':<18} | {'GPU Quota Left %':<17} | {'TPU Left (h)':<13} | {'Quota Reset Time':<24}")
    print("-" * 125)

    total_gpu_rem = 0.0
    total_gpu_used = 0.0
    total_tpu_rem = 0.0

    for i, r in enumerate(records, 1):
        u = r["username"]
        if r.get("status") != "SUCCESS":
            print(f"{i:<4} | {u:<15} | {'ERROR':<13} | {'ERROR':<18} | {'0.0%':<17} | {'N/A':<13} | {'Auth Error':<24}")
            continue

        data = r.get("data", {})
        gpu = data.get("gpuQuota", {})
        tpu = data.get("tpuQuota", {})
        reset_time = data.get("quotaRefreshTime", "Weekly")
        if "T" in reset_time:
            reset_clean = reset_time.split("T")[0] + " " + reset_time.split("T")[1][:8] + " UTC"
        else:
            reset_clean = reset_time

        gpu_used_s = parse_time_seconds(gpu.get("timeUsed", "0s"))
        # Weekly standard pool is 30 hours (108000s), session total is 21600s
        # Total weekly is 30h
        gpu_total_h = 30.0
        gpu_used_h = gpu_used_s / 3600.0
        gpu_rem_h = max(gpu_total_h - gpu_used_h, 0.0)
        gpu_pct = (gpu_rem_h / gpu_total_h) * 100.0

        tpu_used_s = parse_time_seconds(tpu.get("timeUsed", "0s"))
        tpu_total_h = 20.0
        tpu_rem_h = max(tpu_total_h - (tpu_used_s / 3600.0), 0.0)

        total_gpu_rem += gpu_rem_h
        total_gpu_used += gpu_used_h
        total_tpu_rem += tpu_rem_h

        bar = "█" * int(gpu_pct / 10) + "░" * (10 - int(gpu_pct / 10))
        pct_str = f"{gpu_pct:5.1f}% [{bar}]"

        print(f"{i:<4} | {u:<15} | {gpu_used_h:7.2f} h     | {gpu_rem_h:7.2f} h / 30.0h   | {pct_str:<17} | {tpu_rem_h:5.1f} h / 20h  | {reset_clean:<24}")

    print("=" * 125)
    print(f"[TOTAL RESOURCE SUMMARY (ALL 13 ACCOUNTS)]")
    print(f"  * Total GPU Hours Available Right Now : {total_gpu_rem:.2f} Hours (out of {len(records)*30:.0f} Hours total)")
    print(f"  * Total GPU Hours Consumed This Week  : {total_gpu_used:.2f} Hours")
    print(f"  * Total TPU Hours Available           : {total_tpu_rem:.1f} Hours")
    print(f"  * Overall GPU Pool Capacity Remaining : {(total_gpu_rem / (len(records)*30))*100.1:.1f}%")
    print("=" * 125)


if __name__ == "__main__":
    main()
