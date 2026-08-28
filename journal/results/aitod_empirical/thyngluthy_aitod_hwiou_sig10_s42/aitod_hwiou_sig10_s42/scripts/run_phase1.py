"""
Phase 1: Run all baselines sequentially.

Usage:
    python scripts/run_phase1.py                          # run all
    python scripts/run_phase1.py --skip-yolo              # skip YOLO
    python scripts/run_phase1.py --skip-patches           # skip patch baseline
    python scripts/run_phase1.py --seeds 42 123           # custom seeds
"""
from __future__ import annotations
import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT
PROJECT = ROOT.parent


def run(cmd: list[str], desc: str) -> bool:
    print(f"\n{'='*70}")
    print(f"RUNNING: {desc}")
    print(f"CMD: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*70}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT))
    elapsed = time.time() - t0
    ok = result.returncode == 0
    status = "OK" if ok else f"FAILED (code {result.returncode})"
    print(f"\n  -> {status} in {elapsed/60:.1f} min\n")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Run all Phase 1 baselines")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 2024])
    parser.add_argument("--skip-yolo", action="store_true")
    parser.add_argument("--skip-patches", action="store_true")
    parser.add_argument("--yolo-epochs", type=int, default=50)
    parser.add_argument("--yolo-imgsz", type=int, default=640)
    parser.add_argument("--yolo-batch", type=int, default=16)
    args = parser.parse_args()

    print("=" * 70)
    print("PHASE 1 — BASELINE TRAINING")
    print(f"  Seeds: {args.seeds}")
    print(f"  YOLO epochs: {args.yolo_epochs}, imgsz: {args.yolo_imgsz}")
    print("=" * 70)

    all_ok = True

    # ── 1. YOLO ──
    if not args.skip_yolo:
        for model in ["yolov8n.pt", "yolo11n.pt"]:
            ok = run([
                sys.executable, str(SCRIPTS / "train_yolo.py"),
                "--model", model,
                "--epochs", str(args.yolo_epochs),
                "--imgsz", str(args.yolo_imgsz),
                "--batch", str(args.yolo_batch),
            ], f"YOLO baseline: {model}")
            all_ok = all_ok and ok

    # ── 2. FRCNN full-image ──
    for seed in args.seeds:
        ok = run([
            sys.executable, str(SCRIPTS / "train_frcnn_baseline.py"),
            "--seed", str(seed),
        ], f"FRCNN full-image seed={seed}")
        all_ok = all_ok and ok

    # ── 3. FRCNN patches ──
    if not args.skip_patches:
        for seed in args.seeds:
            ok = run([
                sys.executable, str(SCRIPTS / "train_frcnn_baseline.py"),
                "--use-patches", "--seed", str(seed),
            ], f"FRCNN patches seed={seed}")
            all_ok = all_ok and ok

    # ── Summary ──
    print(f"\n{'='*70}")
    if all_ok:
        print("PHASE 1 COMPLETE — all baselines finished successfully")
    else:
        print("PHASE 1 COMPLETE — some runs FAILED (check output above)")
    print(f"{'='*70}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
