from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ALLOWED_EVIDENCE_STATUSES = {
    "diagnostic_invalid_protocol",
    "diagnostic_valid_but_test_reused",
    "validation_evidence",
    "submission_evidence",
}
ALLOWED_CLAIM_STATUSES = {"pending", "enabled", "disabled", "forbidden"}


def read_csv(name: str) -> list[dict[str, str]]:
    path = ROOT / name
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise AssertionError(f"{name} has no rows")
    return rows


def require_unique(rows: list[dict[str, str]], key: str, name: str) -> None:
    values = [row[key].strip() for row in rows]
    if any(not value for value in values):
        raise AssertionError(f"{name} contains an empty {key}")
    if len(values) != len(set(values)):
        raise AssertionError(f"{name} contains duplicate {key} values")


def main() -> None:
    evidence = read_csv("evidence_ledger.csv")
    claims = read_csv("claims_ledger.csv")
    require_unique(evidence, "evidence_id", "evidence_ledger.csv")
    require_unique(claims, "claim_id", "claims_ledger.csv")

    bad_evidence_statuses = sorted(
        {row["status"] for row in evidence} - ALLOWED_EVIDENCE_STATUSES
    )
    if bad_evidence_statuses:
        raise AssertionError(f"invalid evidence statuses: {bad_evidence_statuses}")

    bad_claim_statuses = sorted(
        {row["status"] for row in claims} - ALLOWED_CLAIM_STATUSES
    )
    if bad_claim_statuses:
        raise AssertionError(f"invalid claim statuses: {bad_claim_statuses}")

    for row in evidence:
        if row["paper_a_eligible"] not in {"yes", "no"}:
            raise AssertionError(
                f"{row['evidence_id']} has invalid paper_a_eligible value"
            )
        if not row["reason"].strip():
            raise AssertionError(f"{row['evidence_id']} has no disposition reason")
        if row["status"] == "submission_evidence" and row["paper_a_eligible"] != "yes":
            raise AssertionError(
                f"{row['evidence_id']} is submission evidence but is not eligible"
            )

    for row in claims:
        if not row["evidence_requirement"].strip():
            raise AssertionError(f"{row['claim_id']} has no evidence requirement")
        if not row["required_artifacts"].strip():
            raise AssertionError(f"{row['claim_id']} has no required artifact")
        if row["claim_id"].startswith("F") and row["status"] != "forbidden":
            raise AssertionError(f"{row['claim_id']} must remain forbidden at G0")

    submission_rows = [
        row for row in evidence if row["status"] == "submission_evidence"
    ]
    if submission_rows:
        raise AssertionError("G0 must not promote historical submission evidence")

    print(
        "G0 PASS: "
        f"{len(claims)} claims registered, "
        f"{len(evidence)} evidence families classified, "
        "0 submission-evidence rows"
    )


if __name__ == "__main__":
    main()

