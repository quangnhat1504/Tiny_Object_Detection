from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


RUNS = {
    "smooth_l1_ap75": {
        "env": {
            "TOD_USE_COPY_PASTE": None,
            "TOD_TINY_TILE_OVERSAMPLE": None,
            "TOD_COPY_PASTE_PROB": None,
            "TOD_COPY_PASTE_MAX_PER": None,
        },
        "train": [
            "scripts/train_frcnn_metric.py",
            "--metric", "sa_alw_full",
            "--placement", "la_loss",
            "--seed", "42",
            "--box-loss", "smooth_l1",
            "--tag", "smooth_l1_ap75",
        ],
        "ckpt": "runs/sa_alw_full__smooth_l1__la_loss__seed42__smooth_l1_ap75/best_ap75.pt",
        "analysis": "runs/ap75_analysis_smooth_l1_valid",
    },
    "os1": {
        "env": {
            "TOD_USE_COPY_PASTE": "1",
            "TOD_TINY_TILE_OVERSAMPLE": "1.0",
            "TOD_COPY_PASTE_PROB": None,
            "TOD_COPY_PASTE_MAX_PER": None,
        },
        "train": [
            "scripts/train_frcnn_metric.py",
            "--metric", "sa_alw_full",
            "--placement", "la_loss",
            "--seed", "42",
            "--tag", "os1",
        ],
        "ckpt": "runs/sa_alw_full__la_loss__seed42__os1/best_ap75.pt",
        "analysis": "runs/ap75_analysis_os1_valid",
    },
    "os125": {
        "env": {
            "TOD_USE_COPY_PASTE": "1",
            "TOD_TINY_TILE_OVERSAMPLE": "1.25",
            "TOD_COPY_PASTE_PROB": None,
            "TOD_COPY_PASTE_MAX_PER": None,
        },
        "train": [
            "scripts/train_frcnn_metric.py",
            "--metric", "sa_alw_full",
            "--placement", "la_loss",
            "--seed", "42",
            "--tag", "os125",
        ],
        "ckpt": "runs/sa_alw_full__la_loss__seed42__os125/best_ap75.pt",
        "analysis": "runs/ap75_analysis_os125_valid",
    },    "cp_light": {
        "env": {
            "TOD_USE_COPY_PASTE": "1",
            "TOD_TINY_TILE_OVERSAMPLE": "2.0",
            "TOD_COPY_PASTE_PROB": "0.10",
            "TOD_COPY_PASTE_MAX_PER": "1",
        },
        "train": [
            "scripts/train_frcnn_metric.py",
            "--metric", "sa_alw_full",
            "--placement", "la_loss",
            "--seed", "42",
            "--tag", "cp_light",
        ],
        "ckpt": "runs/sa_alw_full__la_loss__seed42__cp_light/best_ap75.pt",
        "analysis": "runs/ap75_analysis_cp_light_valid",
    },
}


def run(cmd: list[str], env: dict[str, str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], cwd=ROOT, env=env, check=True)


def build_env(args: argparse.Namespace, run_cfg: dict) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env["CPV_DATA_ROOT"] = args.data_root
    for key, value in run_cfg["env"].items():
        if value is None:
            env.pop(key, None)
        else:
            env[key] = value
    return env


def analyze(ckpt: str, out_dir: str, env: dict[str, str]) -> None:
    run([
        "scripts/analyze_ap75_errors.py",
        "--ckpt", ckpt,
        "--split", "valid",
        "--score-thr", "0.05",
        "--topk", "100",
        "--out-dir", out_dir,
    ], env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle runner for tiny-object-detection experiments")
    parser.add_argument("--run", choices=sorted(RUNS), default="smooth_l1_ap75")
    parser.add_argument("--data-root", default="/kaggle/input/datasets/ngquangnht/tinydataset-yolostandard")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-analysis", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"Data root does not exist: {data_root}")

    run_cfg = RUNS[args.run]
    env = build_env(args, run_cfg)

    train_cmd = list(run_cfg["train"])
    if args.resume and "--resume" not in train_cmd:
        train_cmd.append("--resume")

    if not args.skip_train:
        run(train_cmd, env)

    ckpt = ROOT / run_cfg["ckpt"]
    if not args.skip_analysis:
        if not ckpt.exists():
            raise SystemExit(f"Checkpoint not found for analysis: {ckpt}")
        analyze(run_cfg["ckpt"], run_cfg["analysis"], env)

    print("\nKaggle run complete.")


if __name__ == "__main__":
    main()
