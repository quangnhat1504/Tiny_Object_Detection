"""Create the two immutable WP03 A3 seed-42 Kaggle script packages.

This tool is intentionally prepare-only. It never calls Kaggle and never
changes a result ledger. A separate capacity check and explicit push are
required after it has produced and locally validated the two packages.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / ".runtime" / "kaggle" / "wp03" / "a3_seed42"
ACCOUNT = "ngquangnht"
DATASET_SOURCES = [
    "ngquangnht/tinyperson-wp01-a1",
    "ngquangnht/paper-a-code-wp02",
]
METHODS = ("alw_canonical", "sa_alw_full")
SEED = 42
EPOCHS = 8
BATCH_SIZE = 4
TRAINER_SHA256 = "7c05831cbc544b84926694ecdd85159a9ac85ee557a7dc6894bebcfaed2b5d03"
TRAIN_HASH = "5bea11d2d6c4f0e524455d7394492eff85991cb6140987573e8890806f9f026b"
VAL_HASH = "31d67f94a62d3d9ecbbf825a9dca0a21b22b1a297645dfc34402c59cab50ab27"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def kernel_slug(method: str) -> str:
    return f"wp03-a3-{method.replace('_', '-')}-s{SEED}"


def render_kernel(method: str) -> str:
    return f'''"""WP03 A3 seed-42 validation-only kernel: {method}."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
DATA_SLUG = "tinyperson-wp01-a1"
CODE_SLUG = "paper-a-code-wp02"
METHOD = "{method}"
SEED = {SEED}
EPOCHS = {EPOCHS}
BATCH_SIZE = {BATCH_SIZE}
EXPECTED_TRAINER_SHA256 = "{TRAINER_SHA256}"
EXPECTED_TRAIN_HASH = "{TRAIN_HASH}"
EXPECTED_VAL_HASH = "{VAL_HASH}"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_mount(slug):
    candidates = [INPUT_ROOT / slug]
    candidates.extend(INPUT_ROOT.glob(f"datasets/*/{{slug}}"))
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    for candidate in INPUT_ROOT.rglob(slug):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"dataset mount not found: {{slug}}")


def preflight():
    data_root = resolve_mount(DATA_SLUG)
    code_root = resolve_mount(CODE_SLUG)
    import torch
    train_ann = data_root / "splits" / "tinyperson_train_sub.json"
    val_ann = data_root / "splits" / "tinyperson_val.json"
    trainer = code_root / "paper_a" / "tools" / "train_tinyperson_pilot.py"
    report = {{
        "work_package": "WP03_A3",
        "method_invocation_label": METHOD,
        "canonical_method": "sa_alw_canonical" if METHOD == "sa_alw_full" else METHOD,
        "seed": SEED,
        "test_access": "none",
        "data_root": str(data_root),
        "code_root": str(code_root),
        "cuda_available": torch.cuda.is_available(),
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else [],
        "trainer_sha256": sha256(trainer),
        "train_annotation_sha256": sha256(train_ann),
        "val_annotation_sha256": sha256(val_ann),
    }}
    (WORKING / "mount_preflight.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["cuda_available"] or not any("Tesla T4" in name for name in report["gpus"]):
        raise RuntimeError(f"WP03 A3 requires Tesla T4, got {{report['gpus']}}")
    if report["trainer_sha256"] != EXPECTED_TRAINER_SHA256:
        raise RuntimeError("frozen trainer hash mismatch")
    if report["train_annotation_sha256"] != EXPECTED_TRAIN_HASH or report["val_annotation_sha256"] != EXPECTED_VAL_HASH:
        raise RuntimeError("frozen split hash mismatch")
    return data_root, code_root


def main():
    data_root, code_root = preflight()
    torch_cache = Path("/kaggle/tmp/torch_cache")
    if not torch_cache.exists():
        shutil.copytree(code_root / "torch_cache", torch_cache)
    os.environ["TORCH_HOME"] = str(torch_cache)
    os.environ["PAPER_A_TINYPERSON_EVALUATOR"] = str(code_root / "pinned_evaluator" / "tinyperson_cocoeval.py")
    run_root = WORKING / "runs"
    command = [
        sys.executable, "-u", str(code_root / "paper_a" / "tools" / "train_tinyperson_pilot.py"),
        "--method", METHOD, "--seed", str(SEED),
        "--data-root", str(data_root), "--splits-dir", str(data_root / "splits"),
        "--schedule", str(code_root / "paper_a" / "schedules" / "tinyperson_train_p10_p90.json"),
        "--epochs", str(EPOCHS), "--batch-size", str(BATCH_SIZE), "--num-workers", "0",
        "--output-root", str(run_root), "--tag", "wp03_a3",
    ]
    print("launch:", " ".join(command))
    completed = subprocess.run(command, cwd=str(code_root))
    run_dir = run_root / f"wp01_pilot_{{METHOD}}__seed{{SEED}}__wp03_a3"
    if (run_dir / "results.json").exists():
        results = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
        (WORKING / "summary.json").write_text(json.dumps({{
            "method_invocation_label": METHOD,
            "canonical_method": "sa_alw_canonical" if METHOD == "sa_alw_full" else METHOD,
            "seed": SEED,
            "test_access": results.get("test_access"),
            "best_epoch": results.get("best_epoch"),
            "reloaded_evaluation": results.get("reloaded_evaluation"),
        }}, indent=2), encoding="utf-8")
    if completed.returncode:
        raise SystemExit(completed.returncode)
    print("WP03_A3_TRAIN_OK")


if __name__ == "__main__":
    main()
'''


def metadata(method: str) -> dict[str, object]:
    slug = kernel_slug(method)
    return {
        "id": f"{ACCOUNT}/{slug}",
        "title": slug,
        "code_file": "kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": False,
        "dataset_sources": DATASET_SOURCES,
    }


def build_plan() -> dict[str, object]:
    local_trainer = ROOT / "paper_a" / "tools" / "train_tinyperson_pilot.py"
    if sha256(local_trainer) != TRAINER_SHA256:
        raise ValueError("local trainer no longer matches frozen WP02 hash")
    packages = []
    for method in METHODS:
        source = render_kernel(method)
        compile(source, f"{kernel_slug(method)}/kernel.py", "exec")
        packages.append({
            "method_invocation_label": method,
            "canonical_method": "sa_alw_canonical" if method == "sa_alw_full" else method,
            "seed": SEED,
            "kernel": f"{ACCOUNT}/{kernel_slug(method)}",
            "dataset_sources": DATASET_SOURCES,
            "expected_output": f"runs/wp01_pilot_{method}__seed42__wp03_a3/",
            "kernel_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
        })
    return {
        "work_package": "WP03_A3",
        "owner": "Codex (user-authorized)",
        "account": ACCOUNT,
        "accelerator_request": "NvidiaTeslaT4",
        "test_access": "none",
        "trainer_sha256": TRAINER_SHA256,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "packages": packages,
        "status": "READY_PENDING_QUOTA",
    }


def write(plan: dict[str, object]) -> None:
    for method in METHODS:
        directory = STAGE / "kernels" / kernel_slug(method)
        if directory.exists():
            raise ValueError(f"refusing to overwrite existing package: {directory}")
        directory.mkdir(parents=True)
        kernel_path = directory / "kernel.py"
        kernel_path.write_text(render_kernel(method), encoding="utf-8")
        py_compile.compile(str(kernel_path), doraise=True)
        (directory / "kernel-metadata.json").write_text(
            json.dumps(metadata(method), indent=2) + "\n", encoding="utf-8"
        )
    STAGE.mkdir(parents=True, exist_ok=True)
    (STAGE / "run_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true", help="create new package directories")
    args = parser.parse_args()
    plan = build_plan()
    if args.write:
        write(plan)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
