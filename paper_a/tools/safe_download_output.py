"""
Safely download and extract Kaggle kernel output zip avoiding MAX_PATH (WinError 206).
"""
from __future__ import annotations
import io
import json
import os
import sys
import zipfile
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")

def safe_download_kernel_output(ref: str, cred_file: str, dest_dir: Path):
    creds = json.loads((Path.home() / ".kaggle" / cred_file).read_text(encoding="utf-8"))
    os.environ["KAGGLE_USERNAME"] = creds["username"]
    os.environ["KAGGLE_KEY"] = creds["key"]
    api = KaggleApi()
    api.authenticate()

    owner, kernel_slug = ref.split("/")
    dest_dir.mkdir(parents=True, exist_ok=True)

    print(f"Fetching kernel output archive for {ref}...")
    response = api.process_response(
        api.kernel_output_with_http_info(owner, kernel_slug)
    )
    
    # Process zip stream in memory or short temp path
    with zipfile.ZipFile(io.BytesIO(response.data)) as zf:
        for member in zf.infolist():
            # Only extract runs/ and log files or top-level metrics
            fname = member.filename
            if "runs/" in fname or fname.endswith(".log") or fname.endswith(".csv") or fname.endswith(".json") or fname.endswith(".pt"):
                # Clean path
                target_path = Path(f"\\\\?\\{dest_dir.resolve() / fname}")
                if member.is_dir():
                    target_path.mkdir(parents=True, exist_ok=True)
                else:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as src, open(str(target_path), "wb") as dst:
                        dst.write(src.read())
                print(f"  Extracted: {fname}")

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding='utf-8')
    ref = sys.argv[1]
    cred = sys.argv[2]
    out_path = Path(sys.argv[3])
    safe_download_kernel_output(ref, cred, out_path)
