"""
Phase 2: Run all metric ablation experiments sequentially.

Usage:
    python scripts/run_phase2.py                    # run all 21 experiments
    python scripts/run_phase2.py --quick            # 1 seed only (for testing)
    python scripts/run_phase2.py --metric nwd       # single metric, all seeds
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

# Phase 2 configs (plan.md 2.2 → 2.8)
# 2.1 is Phase 1 baseline, no need to re-run
METRICS = [
    "nwd",                # 2.2
    "igwd",               # 2.3
    "igwd_log_shape",     # 2.4
    "igwd_anisotropic_s", # 2.5
    "alw_full",           # 2.6
    "sa_alw_beta_only",   # 2.7
    "sa_alw_full",        # 2.8
]
PLACEMENT = "la_loss"
SEEDS = [42]  # Phase 2: 1 seed per metric for quick screening


def run(cmd: list[str], desc: str) -> bool:
    print(f"\n{'='*70}")
    print(f"RUNNING: {desc}")
    print(f"{'='*70}\n")
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(PROJECT))
    elapsed = time.time() - t0
    ok = result.returncode == 0
    print(f"\n  -> {'OK' if ok else f'FAILED (code {result.returncode})'} in {elapsed/60:.1f} min\n")
    return ok


def main():
    parser = argparse.ArgumentParser(description="Phase 2: metric ablation")
    parser.add_argument("--quick", action="store_true",
                        help="1 seed only (42) for quick test")
    parser.add_argument("--metric", type=str, default=None,
                        help="Run only this metric")
    args = parser.parse_args()

    metrics = [args.metric] if args.metric else METRICS
    seeds = [42] if args.quick else SEEDS

    total = len(metrics) * len(seeds)
    print("=" * 70)
    print(f"PHASE 2 — METRIC CHAIN ABLATION")
    print(f"  Placement: {PLACEMENT}")
    print(f"  Metrics: {len(metrics)}")
    print(f"  Seeds: {seeds}")
    print(f"  Total experiments: {total}")
    print("=" * 70)

    ok_count = 0
    for metric in metrics:
        for seed in seeds:
            ok = run([
                sys.executable, str(SCRIPTS / "train_frcnn_metric.py"),
                "--metric", metric,
                "--placement", PLACEMENT,
                "--seed", str(seed),
            ], f"{metric} @ {PLACEMENT} seed={seed}")
            if ok:
                ok_count += 1

    print(f"\n{'='*70}")
    print(f"PHASE 2 COMPLETE: {ok_count}/{total} succeeded")
    print(f"{'='*70}")
    return 0 if ok_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
