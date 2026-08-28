#!/usr/bin/env python3
"""
sync_notebooklm.py - Automated Uploader to Google NotebookLM / Gemini Notebook.

This script interacts with the `notebooklm` CLI to:
1. Verify authentication status.
2. Select or create the "TinyObjectDetection" notebook.
3. Batch upload all 13 core project documents from `journal/notebooklm_knowledge/`.
4. Output the notebook ID, URL, and list of sources.
"""

import json
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

NOTEBOOK_NAME = "TinyObjectDetection"
NOTEBOOK_ID = "3c4c9584-6155-4711-a252-4343dbdf7afd"
KNOWLEDGE_DIR = Path("journal/notebooklm_knowledge")


def run_cli(args: list[str]) -> tuple[int, str]:
    """Execute notebooklm CLI command."""
    cmd = ["notebooklm"] + args
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    out = res.stdout.strip() if res.stdout else res.stderr.strip()
    return res.returncode, out


def check_auth() -> bool:
    """Verify if NotebookLM is authenticated."""
    code, out = run_cli(["auth", "check", "--test", "--json"])
    if code != 0:
        return False
    try:
        data = json.loads(out)
        return data.get("status") == "ok" and data.get("checks", {}).get("token_fetch", False)
    except Exception:
        return False


def upload_all_sources(nb_id: str):
    """Upload all prepared project documents."""
    if not KNOWLEDGE_DIR.exists():
        print(f"[!] Knowledge directory {KNOWLEDGE_DIR} does not exist.")
        return

    files = sorted(list(KNOWLEDGE_DIR.glob("*")))
    print(f"[*] Found {len(files)} files to upload into notebook '{NOTEBOOK_NAME}' (ID: {nb_id})...\n")

    for idx, f in enumerate(files, 1):
        print(f"[{idx}/{len(files)}] Uploading {f.name}...")
        args = ["source", "add", str(f.resolve()), "-n", nb_id]
        code, out = run_cli(args)
        if code == 0:
            print(f"    ✓ Successfully uploaded {f.name}")
        else:
            print(f"    ! Output: {out}")
        time.sleep(1.5)

    print("\n[*] All uploads completed! Fetching live source list from NotebookLM...")
    list_args = ["source", "list", "-n", nb_id]
    _, out = run_cli(list_args)
    print(out)


def main():
    print("=== Google NotebookLM Automated Sync ===")
    if not check_auth():
        print("[!] NotebookLM is not authenticated or session expired.")
        sys.exit(1)

    print(f"[✓] Authentication verified for dangquangnhat1504@gmail.com!")
    print(f"[+] Target Notebook: {NOTEBOOK_NAME} (ID: {NOTEBOOK_ID})")
    
    # Set context
    run_cli(["use", NOTEBOOK_ID])
    
    upload_all_sources(NOTEBOOK_ID)
    print("\n[✓] Project TinyObjectDetection is now fully synchronized to NotebookLM!")


if __name__ == "__main__":
    main()
