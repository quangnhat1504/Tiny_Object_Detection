from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


REQUIRED_COLUMNS = {
    "shard_id",
    "stage",
    "dataset",
    "source_work_packages",
    "seed",
    "methods",
    "planned_training_runs",
    "atomic_match_group",
    "estimated_gpu_hours",
    "owner",
    "kaggle_account",
    "status",
    "test_access",
    "dependency",
}
ASSIGNED_STATES = {
    "READY_FOR_PUSH",
    "RUNNING",
    "ARTIFACT_AUDIT",
    "ACCEPTED",
    "REJECTED",
    "FAILED",
}
ALLOWED_TEST_ACCESS = {"validation_only", "no_new_test_access"}
EXPECTED_CORE = {
    "TinyPerson": {
        "standard",
        "RFLA",
        "NWD",
        "predecessor",
        "ALW",
        "SA-ALW",
    },
    "AI-TOD-v2": {"standard", "predecessor", "ALW", "SA-ALW"},
}


def _tokens(value: str) -> set[str]:
    return {item.strip() for item in value.split(";") if item.strip()}


def load_rows(path: Path) -> tuple[list[dict[str, str]], set[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), set(reader.fieldnames or [])


def validate_rows(
    rows: Iterable[dict[str, str]],
    fieldnames: set[str],
    *,
    enforce_balance: bool = False,
) -> dict[str, object]:
    errors: list[str] = []
    missing_columns = sorted(REQUIRED_COLUMNS - fieldnames)
    if missing_columns:
        errors.append(f"missing columns: {', '.join(missing_columns)}")

    seen: set[str] = set()
    rows_by_id: dict[str, dict[str, str]] = {}
    owner_hours: defaultdict[str, float] = defaultdict(float)
    assigned_training_rows = 0
    unassigned_training_rows = 0
    core_seeds: defaultdict[str, set[str]] = defaultdict(set)

    for index, row in enumerate(rows, start=2):
        shard_id = row.get("shard_id", "").strip()
        if not shard_id:
            errors.append(f"row {index}: empty shard_id")
            continue
        if shard_id in seen:
            errors.append(f"row {index}: duplicate shard_id {shard_id}")
        seen.add(shard_id)
        rows_by_id[shard_id] = row

        status = row.get("status", "").strip()
        owner = row.get("owner", "").strip()
        account = row.get("kaggle_account", "").strip()
        gpu_hours = row.get("estimated_gpu_hours", "").strip()
        test_access = row.get("test_access", "").strip()
        atomic_group = row.get("atomic_match_group", "").strip()
        planned_runs = row.get("planned_training_runs", "").strip()

        if not atomic_group:
            errors.append(f"{shard_id}: atomic_match_group is required")
        if test_access not in ALLOWED_TEST_ACCESS:
            errors.append(
                f"{shard_id}: ordinary shard test_access must be one of "
                f"{sorted(ALLOWED_TEST_ACCESS)}, got {test_access!r}"
            )

        assigned = owner not in {"", "TBD"} or account not in {"", "TBD"}
        if (owner not in {"", "TBD"}) != (account not in {"", "TBD"}):
            errors.append(f"{shard_id}: owner and kaggle_account must be assigned together")

        numeric_hours: float | None = None
        if gpu_hours not in {"", "TBD"}:
            try:
                numeric_hours = float(gpu_hours)
                if numeric_hours <= 0:
                    raise ValueError
            except ValueError:
                errors.append(f"{shard_id}: estimated_gpu_hours must be positive or TBD")

        if status in ASSIGNED_STATES:
            if not assigned:
                errors.append(f"{shard_id}: {status} requires owner and account")
            if numeric_hours is None and planned_runs != "0":
                errors.append(f"{shard_id}: {status} requires numeric GPU-hours")

        if planned_runs != "0":
            if assigned:
                assigned_training_rows += 1
                if numeric_hours is not None:
                    owner_hours[owner] += numeric_hours
            else:
                unassigned_training_rows += 1

        if shard_id.startswith("CORE-"):
            dataset = row.get("dataset", "").strip()
            methods = _tokens(row.get("methods", ""))
            expected = EXPECTED_CORE.get(dataset)
            if expected is None or methods != expected:
                errors.append(f"{shard_id}: core method set does not match {dataset}")
            core_seeds[dataset].add(row.get("seed", "").strip())

    for dataset in EXPECTED_CORE:
        if core_seeds[dataset] != {"42", "123", "2024"}:
            errors.append(
                f"{dataset}: core seeds must be 42, 123, 2024; "
                f"got {sorted(core_seeds[dataset])}"
            )

    main_pilot = rows_by_id.get("PILOT-D1-S42")
    component_pilot = rows_by_id.get("PILOT-COMP-D1-S42")
    if main_pilot is None or component_pilot is None:
        errors.append("both PILOT-D1-S42 and PILOT-COMP-D1-S42 are required")
    else:
        main_assignment = (
            main_pilot.get("owner", "").strip(),
            main_pilot.get("kaggle_account", "").strip(),
        )
        component_assignment = (
            component_pilot.get("owner", "").strip(),
            component_pilot.get("kaggle_account", "").strip(),
        )
        both_assigned = all(value not in {"", "TBD"} for value in main_assignment)
        both_assigned = both_assigned and all(
            value not in {"", "TBD"} for value in component_assignment
        )
        if both_assigned and main_assignment != component_assignment:
            errors.append(
                "PILOT-COMP-D1-S42 must match PILOT-D1-S42 owner and account"
            )

    balance_ratio: float | None = None
    if owner_hours:
        mean = sum(owner_hours.values()) / len(owner_hours)
        balance_ratio = max(owner_hours.values()) / mean
        if enforce_balance and unassigned_training_rows == 0 and balance_ratio > 1.15:
            errors.append(
                f"assigned GPU-hour load exceeds 1.15x team mean: {balance_ratio:.4f}"
            )

    return {
        "status": "PASS" if not errors else "FAIL",
        "shards": len(seen),
        "assigned_training_shards": assigned_training_rows,
        "unassigned_training_shards": unassigned_training_rows,
        "owner_gpu_hours": dict(sorted(owner_hours.items())),
        "max_to_mean_gpu_hour_ratio": balance_ratio,
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("paper_a/experiments/team_run_shards.csv"),
    )
    parser.add_argument("--enforce-balance", action="store_true")
    args = parser.parse_args()
    rows, fieldnames = load_rows(args.path)
    result = validate_rows(rows, fieldnames, enforce_balance=args.enforce_balance)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
