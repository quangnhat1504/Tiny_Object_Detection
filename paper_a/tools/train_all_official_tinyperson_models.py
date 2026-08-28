"""Train all official TinyPerson 1-class models sequentially."""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable

METHODS = [
    ("standard", "Faster R-CNN Baseline (Standard IoU)"),
    ("nwd", "NWD (NeurIPS 2021)"),
    ("igwd", "IGWD (IEEE TMM 2022)"),
    ("sa_alw_full", "SA-ALW (Paper A)"),
    ("h_wiou", "Homotopy Wasserstein-IoU (Proposed Ours)"),
]

def main():
    parser = argparse.ArgumentParser(description="Train all official TinyPerson models")
    parser.add_argument("--epochs", type=int, default=12, help="Number of epochs per model (default: 12)")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: 2)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--output-root", type=str, default="runs/official_tinyperson_runs", help="Output directory")
    parser.add_argument("--methods", nargs="+", default=["standard", "nwd", "igwd", "sa_alw_full", "h_wiou"], help="Methods to train")
    args = parser.parse_args()

    out_root = Path(args.output_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("=== STARTING OFFICIAL TINYPERSON BENCHMARK TRAINING RUNS ===")
    print(f"Epochs: {args.epochs} | Batch Size: {args.batch_size} | Seed: {args.seed}")
    print(f"Output Directory: {out_root.resolve()}")
    print("=" * 80)

    start_total = time.time()

    for method in args.methods:
        print("\n" + "#" * 80)
        print(f"--> TRAINING METHOD: {method.upper()} on Official TinyPerson Dataset")
        print("#" * 80)
        cmd = [
            PYTHON,
            str(ROOT / "paper_a" / "tools" / "train_tinyperson_pilot.py"),
            "--method", method,
            "--epochs", str(args.epochs),
            "--batch-size", str(args.batch_size),
            "--seed", str(args.seed),
            "--output-root", str(out_root),
        ]
        res = subprocess.run(cmd, cwd=str(ROOT))
        if res.returncode != 0:
            print(f"[ERROR] Failed training for method {method} with return code {res.returncode}")
        else:
            print(f"[SUCCESS] Finished training method {method}")

    total_time = time.time() - start_total
    print("\n" + "=" * 80)
    print(f"=== ALL TRAINING RUNS COMPLETED IN {total_time/60:.2f} MINUTES ===")
    print("=" * 80)

if __name__ == "__main__":
    main()
