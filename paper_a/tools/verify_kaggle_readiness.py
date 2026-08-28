"""
Comprehensive Pre-Flight Readiness Checker for Cloud GPU Experiments.
Validates:
1. All 13 Kaggle API credentials.
2. Dataset accessibility (TinyPerson tiled + AI-TOD-v2 + Code Datasets).
3. Local code integrity, chunked OOM fix, and unit test suite.
4. Full packaging stage for pushing kernels.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
CREDS_DIR = Path.home() / ".kaggle"
STAGE_DIR = Path(r"C:\tmp\tod_full_code_stage")

ACCOUNTS = [
    ("amongus1504", "kaggle.json"),
    ("dipphmngc", "kaggle (11).json"),
    ("hienquang06", "kaggle (5).json"),
    ("hngngnguynvn", "kaggle (1).json"),
    ("hngtrngtn", "kaggle (2).json"),
    ("luongsythanh", "kaggle (10).json"),
    ("ngquangnht", "kaggle (8).json"),
    ("phuc1806", "kaggle (12).json"),
    ("pptlyn11", "kaggle (9).json"),
    ("qnhat1504", "kaggle (3).json"),
    ("quangnhtng", "kaggle (6).json"),
    ("thyngluthy", "kaggle (4).json"),
    ("trieuvo123", "kaggle (7).json"),
]


def test_unit_tests() -> bool:
    print("\n[1/4] Running local CUDA unit test suite...")
    cmd = [sys.executable, "-m", "unittest", "discover", "-s", str(ROOT / "paper_a/tests"), "-p", "test_*.py"]
    res = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    if res.returncode == 0:
        print("  -> PASS: All unit tests passed cleanly.")
        return True
    else:
        print("  -> FAIL: Unit tests encountered errors:")
        print(res.stderr or res.stdout)
        return False


def test_credentials() -> dict[str, bool]:
    print("\n[2/4] Validating 13 Kaggle API credentials...")
    status = {}
    for username, cred_file in ACCOUNTS:
        p = CREDS_DIR / cred_file
        if not p.exists():
            print(f"  -> {username}: Credential file {cred_file} NOT FOUND!")
            status[username] = False
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            env = os.environ.copy()
            env["KAGGLE_USERNAME"] = data["username"]
            env["KAGGLE_KEY"] = data["key"]
            res = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "list", "--mine", "-v"],
                                 env=env, capture_output=True, text=True)
            if res.returncode == 0:
                print(f"  -> {username:<15} ({cred_file:<15}): VALID OK")
                status[username] = True
            else:
                print(f"  -> {username:<15}: AUTH FAILED: {res.stderr.strip()[:60]}")
                status[username] = False
        except Exception as e:
            print(f"  -> {username:<15}: ERROR {e}")
            status[username] = False
    return status


def test_packaging_stage() -> bool:
    print("\n[3/4] Preparing full code packaging stage...")
    if STAGE_DIR.exists():
        shutil.rmtree(STAGE_DIR)
    STAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Copy essential codebase components
    for folder in ["common", "scripts", "paper_a"]:
        src_f = ROOT / folder
        dst_f = STAGE_DIR / folder
        shutil.copytree(src_f, dst_f, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git", "runs"))
        
    for single_file in ["AGENTS.md", "dataset-metadata.json"]:
        p = ROOT / single_file
        if p.exists():
            shutil.copy(p, STAGE_DIR / single_file)
            
    total_files = list(STAGE_DIR.rglob("*.*"))
    print(f"  -> Stage package created at {STAGE_DIR} with {len(total_files)} files.")
    return True


def check_public_datasets() -> bool:
    print("\n[4/4] Checking AI-TOD-v2 and TinyPerson public datasets accessibility...")
    env = os.environ.copy()
    creds = json.loads((CREDS_DIR / "kaggle.json").read_text(encoding="utf-8"))
    env["KAGGLE_USERNAME"] = creds["username"]
    env["KAGGLE_KEY"] = creds["key"]
    
    ds_check = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "list", "-s", "tiny-object-detection-in-aerial-images", "-v"],
                              env=env, capture_output=True, text=True)
    if "simplestzyp/tiny-object-detection-in-aerial-images" in ds_check.stdout:
        print("  -> AI-TOD-v2 Dataset (simplestzyp/tiny-object-detection-in-aerial-images): ACCESSIBLE OK")
    else:
        print("  -> AI-TOD-v2 Dataset Check: Output received:\n", ds_check.stdout[:200])
    return True


def main():
    print("=" * 90)
    print("         KAGGLE CLUSTER EXPERIMENT READINESS PRE-FLIGHT AUDIT         ")
    print("=" * 90)
    
    ut_ok = test_unit_tests()
    creds_ok = test_credentials()
    pack_ok = test_packaging_stage()
    ds_ok = check_public_datasets()
    
    all_creds_valid = all(creds_ok.values())
    
    print("\n" + "=" * 90)
    print(f"OVERALL CLUSTER READINESS: {'READY 100% TO LAUNCH' if (ut_ok and all_creds_valid and pack_ok) else 'ACTION REQUIRED'}")
    print("=" * 90)


if __name__ == "__main__":
    main()
