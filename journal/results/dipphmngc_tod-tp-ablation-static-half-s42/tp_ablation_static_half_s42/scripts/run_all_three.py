"""
Run all 3 decoupled regression experiments sequentially.
A: SA-ALW + Smooth-L1 (mirrors RFLA AP75=18.8 pattern)
B: SA-ALW + CIoU
C: SA-ALW + DIoU

Usage: python scripts/run_all_three.py
"""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = r"C:\Users\ADMIN\AppData\Local\Programs\Python\Python313\python.exe"

experiments = [
    ("A", "smooth_l1"),
    ("B", "ciou"),
    ("C", "diou"),
]

for name, box_loss in experiments:
    output_dir = ROOT / "runs" / f"sa_alw_full__{box_loss}__la_loss__seed42"

    if (output_dir / "best.pt").exists():
        print(f"[SKIP] Experiment {name}: best.pt already exists -> {output_dir}")
        continue

    cmd = [
        PYTHON, str(ROOT / "scripts" / "train_frcnn_metric.py"),
        "--metric", "sa_alw_full",
        "--placement", "la_loss",
        "--seed", "42",
        "--box-loss", box_loss,
    ]
    if (output_dir / "last.pt").exists():
        cmd.append("--resume")

    print(f"\n{'='*60}")
    print(f"EXPERIMENT {name}: sa_alw_full + {box_loss} regression")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    t0 = time.time()
    result = subprocess.run(cmd, cwd=str(ROOT))
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n[FAIL] Experiment {name} crashed with exit code {result.returncode}")
        sys.exit(result.returncode)

    print(f"\n[DONE] Experiment {name}: completed in {elapsed/60:.1f} min")

print("\n" + "="*60)
print("ALL 3 EXPERIMENTS DONE")
print("Evaluate: python scripts/test_eval.py --metric sa_alw_full")
print("="*60)
