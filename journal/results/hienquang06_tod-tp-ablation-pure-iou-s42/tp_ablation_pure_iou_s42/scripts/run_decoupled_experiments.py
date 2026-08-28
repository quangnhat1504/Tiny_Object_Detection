"""Chain-launch Experiment A→B→C sequentially.

Usage: python scripts/run_decoupled_experiments.py
"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = r"C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe"

experiments = [
    ("A", "smooth_l1", "sa_alw_full", 42),
    ("B", "ciou", "sa_alw_full", 42),
    ("C", "diou", "sa_alw_full", 42),
]

for name, box_loss, metric, seed in experiments:
    output_dir = ROOT / "runs" / f"{metric}__{box_loss}__la_loss__seed{seed}"
    if (output_dir / "best.pt").exists():
        print(f"[SKIP] Experiment {name}: already trained ({output_dir})")
        continue

    print(f"\n{'='*70}")
    print(f"EXPERIMENT {name}: {metric} + {box_loss} regression")
    print(f"{'='*70}")

    cmd = [
        PYTHON, str(ROOT / "scripts" / "train_frcnn_metric.py"),
        "--metric", metric,
        "--placement", "la_loss",
        "--seed", str(seed),
        "--box-loss", box_loss,
    ]
    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"[FAIL] Experiment {name} crashed with code {result.returncode}")
        sys.exit(result.returncode)
    print(f"[DONE] Experiment {name} completed in {elapsed/60:.0f}min")

print("\n" + "="*70)
print("ALL 3 EXPERIMENTS COMPLETED")
print("Run: python scripts/test_eval.py --metric sa_alw_full")
print("="*70)
