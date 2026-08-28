"""Validate Paper A result ledgers before manuscript table generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RUN_MANIFEST = ROOT / "runs" / "manifest.jsonl"
RUN_COLUMNS = [
    "run_id", "dataset", "dataset_version", "split_hash", "detector",
    "backbone", "method", "placement", "seed", "code_commit", "config_hash",
    "train_budget", "checkpoint_rule", "best_epoch", "AP", "AP50", "AP75",
    "APS", "APM", "APL", "AR100", "latency_ms", "vram_mb", "status", "notes",
]
RUN_FILES = [
    "main_results.csv",
    "component_ablation.csv",
    "placement_ablation.csv",
    "sensitivity.csv",
    "efficiency.csv",
]
CI_COLUMNS = [
    "comparison_id", "dataset", "method_a", "method_b", "seeds", "metric",
    "delta_mean", "ci_low", "ci_high", "bootstrap_unit",
    "bootstrap_replicates", "status", "artifact_hash", "notes",
]
REQUIRED_ACCEPTED = [
    "run_id", "dataset", "dataset_version", "split_hash", "detector",
    "backbone", "method", "placement", "seed", "code_commit", "config_hash",
    "train_budget", "checkpoint_rule", "best_epoch", "AP", "AP50", "AP75",
]
METRICS = ["AP", "AP50", "AP75", "APS", "APM", "APL", "AR100"]
ALLOWED_SEEDS = {"42", "123", "2024"}
ALLOWED_STATUS = {"PENDING_AUDIT", "ACCEPTED", "REJECTED"}
FORBIDDEN_METHOD_TOKENS = {"cbl", "icbl", "cascade", "refinement", "distill", "pc_mr", "pc_moc"}


def _manifest_ids() -> set[str]:
    ids: set[str] = set()
    if not RUN_MANIFEST.exists():
        return ids
    for line_number, line in enumerate(RUN_MANIFEST.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"manifest.jsonl:{line_number}: invalid JSON: {error}") from error
        run_id = str(record.get("run_id", "")).strip()
        if not run_id:
            raise ValueError(f"manifest.jsonl:{line_number}: missing run_id")
        if run_id in ids:
            raise ValueError(f"manifest.jsonl:{line_number}: duplicate run_id {run_id}")
        ids.add(run_id)
    return ids


def _read_csv(path: Path, expected_columns: list[str]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != expected_columns:
            raise ValueError(f"{path.name}: schema mismatch: {reader.fieldnames}")
        return list(reader)


def _validate_run_row(path: Path, row_number: int, row: dict[str, str], manifest_ids: set[str]) -> None:
    status = row["status"].strip()
    if status not in ALLOWED_STATUS:
        raise ValueError(f"{path.name}:{row_number}: invalid status {status!r}")
    if status != "ACCEPTED":
        return

    missing = [field for field in REQUIRED_ACCEPTED if not row[field].strip()]
    if missing:
        raise ValueError(f"{path.name}:{row_number}: accepted row missing {missing}")
    if row["run_id"] not in manifest_ids:
        raise ValueError(f"{path.name}:{row_number}: accepted run missing from manifest")
    if row["seed"] not in ALLOWED_SEEDS:
        raise ValueError(f"{path.name}:{row_number}: unregistered seed {row['seed']}")
    if row["checkpoint_rule"] != "coco_ap":
        raise ValueError(f"{path.name}:{row_number}: checkpoint_rule must be coco_ap")
    method = row["method"].lower()
    leaked = sorted(token for token in FORBIDDEN_METHOD_TOKENS if token in method)
    if leaked:
        raise ValueError(f"{path.name}:{row_number}: out-of-scope method tokens {leaked}")
    for metric in METRICS:
        value = row[metric].strip()
        if not value:
            continue
        numeric = float(value)
        if not 0.0 <= numeric <= 1.0:
            raise ValueError(f"{path.name}:{row_number}: {metric} must be a fraction in [0,1]")


def validate() -> dict[str, int | str]:
    manifest_ids = _manifest_ids()
    seen_ids: set[str] = set()
    accepted = 0
    total = 0
    for name in RUN_FILES:
        path = RESULTS / name
        rows = _read_csv(path, RUN_COLUMNS)
        for row_number, row in enumerate(rows, 2):
            total += 1
            run_id = row["run_id"].strip()
            if run_id and run_id in seen_ids:
                raise ValueError(f"{path.name}:{row_number}: duplicate result run_id {run_id}")
            if run_id:
                seen_ids.add(run_id)
            _validate_run_row(path, row_number, row, manifest_ids)
            accepted += row["status"].strip() == "ACCEPTED"

    ci_rows = _read_csv(RESULTS / "bootstrap_ci.csv", CI_COLUMNS)
    for row_number, row in enumerate(ci_rows, 2):
        if row["status"] == "ACCEPTED":
            required = ["comparison_id", "dataset", "method_a", "method_b", "seeds", "metric", "delta_mean", "ci_low", "ci_high", "bootstrap_unit", "bootstrap_replicates", "artifact_hash"]
            missing = [field for field in required if not row[field].strip()]
            if missing:
                raise ValueError(f"bootstrap_ci.csv:{row_number}: accepted row missing {missing}")
            if row["bootstrap_unit"] != "original_image":
                raise ValueError(f"bootstrap_ci.csv:{row_number}: bootstrap unit must be original_image")

    return {"status": "PASS", "run_rows": total, "accepted_rows": accepted, "ci_rows": len(ci_rows)}


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2))
