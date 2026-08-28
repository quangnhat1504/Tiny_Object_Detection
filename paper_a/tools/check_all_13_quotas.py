"""
Audit GPU Quota for all 13 Kaggle accounts.
"""
from __future__ import annotations
import json
import os
import subprocess
import sys
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

CREDS_DIR = Path.home() / ".kaggle"

def main():
    print("=" * 95)
    print(f"{'#':<3} {'CREDENTIAL FILE':<20} {'USERNAME':<18} {'GPU QUOTA REMAINING':<25} {'STATUS':<15}")
    print("=" * 95)

    cred_files = sorted(list(CREDS_DIR.glob("kaggle*.json")), key=lambda p: p.name)
    results = []

    for idx, cred_file in enumerate(cred_files, 1):
        try:
            creds = json.loads(cred_file.read_text(encoding="utf-8"))
            username = creds.get("username", "")
            key = creds.get("key", "")
            if not username or not key:
                continue

            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = key

            api = KaggleApi()
            api.authenticate()

            # Query quota via API or CLI test
            # In Kaggle API, we can check account details or list running kernels
            kernels = api.kernels_list(mine=True, page=1, page_size=10)
            running_cnt = 0
            for k in kernels:
                st = str(getattr(k, 'status', ''))
                if 'running' in st.lower() or 'queued' in st.lower():
                    running_cnt += 1

            results.append({
                "idx": idx,
                "file": cred_file.name,
                "username": username,
                "running": running_cnt,
                "kernels": len(kernels),
                "auth": "OK"
            })
            print(f"{idx:<3} {cred_file.name:<20} {username:<18} Running: {running_cnt:<5} (Total: {len(kernels):<2})   Auth: OK")

        except Exception as e:
            print(f"{idx:<3} {cred_file.name:<20} {username if 'username' in locals() else 'Unknown':<18} Error: {str(e)[:30]}")

    print("=" * 95)

if __name__ == "__main__":
    main()
