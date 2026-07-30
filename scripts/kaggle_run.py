from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


RUNS = {
    "cbl_full": {
        "env": {
            "TOD_USE_COPY_PASTE": "1",
            "TOD_TINY_TILE_OVERSAMPLE": "2.0",
            "TOD_COPY_PASTE_PROB": None,
            "TOD_COPY_PASTE_MAX_PER": None,
        },
        "train": [
            "scripts/train_frcnn_metric.py",
            "--metric", "sa_alw_full",
            "--placement", "la_loss",
            "--seed", "42",
            "--box-loss", "cbl",
            "--box-loss-warmup-epochs", "0",
            "--tag", "cbl_full",
        ],
        "ckpt": "runs/sa_alw_full__cbl__la_loss__seed42__cbl_full/best_ap75.pt",
        "analysis": "runs/ap75_analysis_cbl_full_valid",
    },
    "cbl_ema8": {
        "env": {
            "TOD_EPOCHS": "8",
            "TOD_USE_EMA": "1",
            "TOD_USE_COPY_PASTE": "1",
            "TOD_TINY_TILE_OVERSAMPLE": "2.0",
            "TOD_COPY_PASTE_PROB": None,
            "TOD_COPY_PASTE_MAX_PER": None,
        },
        "train": [
            "scripts/train_frcnn_metric.py",
            "--metric", "sa_alw_full",
            "--placement", "la_loss",
            "--seed", "42",
            "--box-loss", "cbl",
            "--box-loss-warmup-epochs", "0",
            "--tag", "cbl_ema8",
        ],
        "ckpt": "runs/sa_alw_full__cbl__la_loss__seed42__cbl_ema8/best_ap75.pt",
        "analysis": "runs/ap75_analysis_cbl_ema8_valid",
    },
    "cbl_qfl_local_gate": {
        "env": {
            "TOD_EPOCHS": "2",
            "TOD_USE_EMA": "0",
            "TOD_NUM_WORKERS": "0",
            "TOD_USE_COPY_PASTE": "1",
            "TOD_TINY_TILE_OVERSAMPLE": "2.0",
            "TOD_COPY_PASTE_PROB": None,
            "TOD_COPY_PASTE_MAX_PER": None,
        },
        "train": [
            "scripts/train_frcnn_metric.py",
            "--metric", "sa_alw_full",
            "--placement", "la_loss",
            "--seed", "42",
            "--box-loss", "cbl",
            "--box-loss-warmup-epochs", "0",
            "--quality-focal",
            "--quality-focal-beta", "2.0",
            "--tag", "cbl_qfl_local_gate",
        ],
        "ckpt": "runs/sa_alw_full__cbl__qflb2__la_loss__seed42__cbl_qfl_local_gate/best_ap75.pt",
        "analysis": "runs/ap75_analysis_cbl_qfl_local_gate_valid",
    },
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
    "q_smooth_l1_w025": {
        "env": {
            "TOD_USE_COPY_PASTE": "1",
            "TOD_TINY_TILE_OVERSAMPLE": "2.0",
            "TOD_COPY_PASTE_PROB": None,
            "TOD_COPY_PASTE_MAX_PER": None,
        },
        "train": [
            "scripts/train_frcnn_metric.py",
            "--metric", "sa_alw_full",
            "--placement", "la_loss",
            "--seed", "42",
            "--box-loss", "smooth_l1",
            "--quality-score",
            "--quality-loss-weight", "0.25",
            "--tag", "q_smooth_l1_w025",
        ],
        "ckpt": "runs/sa_alw_full__smooth_l1__q0.25__la_loss__seed42__q_smooth_l1_w025/best_ap75.pt",
        "analysis": "runs/ap75_analysis_q_smooth_l1_w025_valid",
    },
    "q_smooth_l1_w05": {
        "env": {
            "TOD_USE_COPY_PASTE": "1",
            "TOD_TINY_TILE_OVERSAMPLE": "2.0",
            "TOD_COPY_PASTE_PROB": None,
            "TOD_COPY_PASTE_MAX_PER": None,
        },
        "train": [
            "scripts/train_frcnn_metric.py",
            "--metric", "sa_alw_full",
            "--placement", "la_loss",
            "--seed", "42",
            "--box-loss", "smooth_l1",
            "--quality-score",
            "--quality-loss-weight", "0.5",
            "--tag", "q_smooth_l1_w05",
        ],
        "ckpt": "runs/sa_alw_full__smooth_l1__q0.5__la_loss__seed42__q_smooth_l1_w05/best_ap75.pt",
        "analysis": "runs/ap75_analysis_q_smooth_l1_w05_valid",
    },
    "q_smooth_l1_w10": {
        "env": {
            "TOD_USE_COPY_PASTE": "1",
            "TOD_TINY_TILE_OVERSAMPLE": "2.0",
            "TOD_COPY_PASTE_PROB": None,
            "TOD_COPY_PASTE_MAX_PER": None,
        },
        "train": [
            "scripts/train_frcnn_metric.py",
            "--metric", "sa_alw_full",
            "--placement", "la_loss",
            "--seed", "42",
            "--box-loss", "smooth_l1",
            "--quality-score",
            "--quality-loss-weight", "1.0",
            "--tag", "q_smooth_l1_w10",
        ],
        "ckpt": "runs/sa_alw_full__smooth_l1__q1__la_loss__seed42__q_smooth_l1_w10/best_ap75.pt",
        "analysis": "runs/ap75_analysis_q_smooth_l1_w10_valid",
    },
    "q_metric_w05": {
        "env": {
            "TOD_USE_COPY_PASTE": "1",
            "TOD_TINY_TILE_OVERSAMPLE": "2.0",
            "TOD_COPY_PASTE_PROB": None,
            "TOD_COPY_PASTE_MAX_PER": None,
        },
        "train": [
            "scripts/train_frcnn_metric.py",
            "--metric", "sa_alw_full",
            "--placement", "la_loss",
            "--seed", "42",
            "--box-loss", "metric",
            "--quality-score",
            "--quality-loss-weight", "0.5",
            "--tag", "q_metric_w05",
        ],
        "ckpt": "runs/sa_alw_full__q0.5__la_loss__seed42__q_metric_w05/best_ap75.pt",
        "analysis": "runs/ap75_analysis_q_metric_w05_valid",
    },
    "q_diou_w05": {
        "env": {
            "TOD_USE_COPY_PASTE": "1",
            "TOD_TINY_TILE_OVERSAMPLE": "2.0",
            "TOD_COPY_PASTE_PROB": None,
            "TOD_COPY_PASTE_MAX_PER": None,
        },
        "train": [
            "scripts/train_frcnn_metric.py",
            "--metric", "sa_alw_full",
            "--placement", "la_loss",
            "--seed", "42",
            "--box-loss", "diou",
            "--quality-score",
            "--quality-loss-weight", "0.5",
            "--tag", "q_diou_w05",
        ],
        "ckpt": "runs/sa_alw_full__diou__q0.5__la_loss__seed42__q_diou_w05/best_ap75.pt",
        "analysis": "runs/ap75_analysis_q_diou_w05_valid",
    },
    "q_smooth_l1_w05_seed2024": {
        "env": {
            "TOD_USE_COPY_PASTE": "1",
            "TOD_TINY_TILE_OVERSAMPLE": "2.0",
            "TOD_COPY_PASTE_PROB": None,
            "TOD_COPY_PASTE_MAX_PER": None,
        },
        "train": [
            "scripts/train_frcnn_metric.py",
            "--metric", "sa_alw_full",
            "--placement", "la_loss",
            "--seed", "2024",
            "--box-loss", "smooth_l1",
            "--quality-score",
            "--quality-loss-weight", "0.5",
            "--tag", "q_smooth_l1_w05_seed2024",
        ],
        "ckpt": "runs/sa_alw_full__smooth_l1__q0.5__la_loss__seed2024__q_smooth_l1_w05_seed2024/best_ap75.pt",
        "analysis": "runs/ap75_analysis_q_smooth_l1_w05_seed2024_valid",
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
