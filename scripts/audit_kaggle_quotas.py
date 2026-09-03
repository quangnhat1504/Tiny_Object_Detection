"""
Audit remaining GPU/TPU hours across all 13 Kaggle accounts.
"""
from __future__ import annotations
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"
TMP_PROFILE_ROOT = Path(r"C:\tmp\tod_kaggle_quota_audit")
TMP_PROFILE_ROOT.mkdir(parents=True, exist_ok=True)

INDEX_FILE = CREDS_DIR / "account_index.json"


def parse_seconds(val: str | float | int) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    # Extract first sequence of digits possibly with decimal point
    m = re.search(r"(\d+(?:\.\d+)?)", str(val))
    if m:
        return float(m.group(1))
    return 0.0


def audit_quotas():
    with open(INDEX_FILE, "r", encoding="utf-8") as f:
        accounts = json.load(f)

    print("=" * 85)
    print(f"{'No.':<4} | {'Username':<15} | {'GPU Used':<12} | {'GPU Total':<12} | {'GPU Rem':<10} | {'Status':<10}")
    print("=" * 85)

    summary = []
    for item in accounts:
        idx = item["order"]
        fname = item["filename"]
        uname = item["username"]

        cred_path = CREDS_DIR / fname
        if not cred_path.exists():
            print(f"{idx:<4} | {uname:<15} | MISSING CREDENTIAL")
            continue

        acc_dir = TMP_PROFILE_ROOT / uname
        acc_dir.mkdir(parents=True, exist_ok=True)
        target_cred = acc_dir / "kaggle.json"
        shutil.copy(cred_path, target_cred)

        env = os.environ.copy()
        env["KAGGLE_CONFIG_DIR"] = str(acc_dir)

        code = (
            "import os, json; from kaggle.api.kaggle_api_extended import KaggleApi; "
            "api = KaggleApi(); api.authenticate(); "
            "q = json.loads(str(api.quota_view())); "
            "gpu = q.get('gpuQuota', {}); "
            "print('RESULT:' + json.dumps(gpu))"
        )
        try:
            res = subprocess.run(
                [sys.executable, "-c", code],
                env=env,
                capture_output=True,
                text=True,
                timeout=15,
            )
            out = res.stdout.strip()
            if "RESULT:" in out:
                gpu_json = out.split("RESULT:")[1].strip()
                gpu_data = json.loads(gpu_json)
                used_s = parse_seconds(gpu_data.get("timeUsed", "0s"))
                total_s = parse_seconds(gpu_data.get("totalTimeAllowed", "108000s"))
                rem_h = max(round((total_s - used_s) / 3600.0, 2), 0.0)
                used_h = round(used_s / 3600.0, 2)
                total_h = round(total_s / 3600.0, 2)
                status = "HEALTHY" if rem_h > 5.0 else ("LOW" if rem_h > 0.5 else "EXHAUSTED")
                print(f"{idx:<4} | {uname:<15} | {used_h:>5.1f}h      | {total_h:>5.1f}h      | {rem_h:>5.1f}h    | {status}")
                summary.append({"order": idx, "username": uname, "rem_h": rem_h, "total_h": total_h, "status": status})
            else:
                err = (res.stderr or res.stdout).strip().replace("\n", " ")[:25]
                print(f"{idx:<4} | {uname:<15} | ERROR: {err}")
                summary.append({"order": idx, "username": uname, "rem_h": 0.0, "status": f"ERROR: {err}"})
        except subprocess.TimeoutExpired:
            print(f"{idx:<4} | {uname:<15} | TIMEOUT (15s)")
            summary.append({"order": idx, "username": uname, "rem_h": 0.0, "status": "TIMEOUT"})
        except Exception as e:
            print(f"{idx:<4} | {uname:<15} | EXCEPTION: {e}")

    print("=" * 85)
    total_avail = sum(s.get("rem_h", 0.0) for s in summary)
    print(f"TOTAL AVAILABLE GPU HOURS ACROSS CLUSTER: {total_avail:.2f} HOURS")
    print("=" * 85)
    return summary


if __name__ == "__main__":
    audit_quotas()
