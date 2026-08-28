"""Validate the immutable evidence proposed for WP03 A2 re-adjudication.

This validator deliberately does not edit result ledgers or make a promotion
decision. It proves only that the four downloaded v12 T4 diagnostic reports
remain bound to the two original v8 canonical-ALW artifacts and satisfy the
unchanged primary-metric tolerance.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / ".runtime" / "kaggle" / "wp03" / "v9_t4_audit"
V8_ROOT = ROOT / ".runtime" / "kaggle" / "wp03" / "v8" / "outputs"
LEDGER_PATH = ROOT / "paper_a" / "protocol_ledger.md"
TOLERANCE = 5e-4
TRAINER_SHA256 = "7c05831cbc544b84926694ecdd85159a9ac85ee557a7dc6894bebcfaed2b5d03"

REPORTS = {
    "s123_r1": ("hngngnguynvn", 123, "r1", "b1fe988f58d865c885498c4dea13ec8308847a4fdd9422a9d1176ca394b28cd2"),
    "s123_r2": ("quangnhtng", 123, "r2", "b1fe988f58d865c885498c4dea13ec8308847a4fdd9422a9d1176ca394b28cd2"),
    "s2024_r1": ("hngtrngtn", 2024, "r1", "806bffbe3c2ebea6a4eeb0aa23f8ea299d5d1702e4297c72409b06d7caacd122"),
    "s2024_r2": ("luongsythanh", 2024, "r2", "806bffbe3c2ebea6a4eeb0aa23f8ea299d5d1702e4297c72409b06d7caacd122"),
}
SOURCE_MANIFESTS = {
    123: AUDIT_ROOT / "datasets" / "quangnhtng" / "wp03-v10-alw-audit-input-s123" / "artifact" / "artifact_manifest.json",
    2024: AUDIT_ROOT / "datasets" / "hngtrngtn" / "wp03-v10-alw-audit-input-s2024" / "artifact" / "artifact_manifest.json",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ValueError(f"missing required artifact: {path}") from error


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate() -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for identifier, (account, seed, replica, expected_manifest) in REPORTS.items():
        report_path = AUDIT_ROOT / "outputs" / "v12" / identifier / "audit_report.json"
        report = load_json(report_path)
        require(report.get("audit_kind") == "diagnostic_only_no_training_no_test", f"{identifier}: invalid audit kind")
        require(report.get("method") == "alw_canonical", f"{identifier}: method is not canonical ALW")
        require(report.get("account") == account, f"{identifier}: account mismatch")
        require(report.get("seed") == seed and report.get("replica") == replica, f"{identifier}: seed/replica mismatch")
        require(report.get("manifest_sha256") == expected_manifest, f"{identifier}: manifest hash mismatch")

        runtime = report.get("runtime", {})
        require(runtime.get("gpu_count") == 2, f"{identifier}: expected two GPUs")
        require(runtime.get("gpus") == ["Tesla T4", "Tesla T4"], f"{identifier}: expected Tesla T4 x2")

        saved = report.get("saved_replay", {})
        regenerated = report.get("regenerated_inference", {})
        for family in ("official_primary_deltas_vs_kernel", "paper_primary_deltas_vs_kernel"):
            deltas = saved.get(family, {})
            require(bool(deltas) and max(deltas.values()) == 0.0, f"{identifier}: saved replay is not exact for {family}")
        for field in ("official_primary_max_delta", "paper_primary_max_delta"):
            value = regenerated.get(field)
            require(isinstance(value, (int, float)) and value <= TOLERANCE, f"{identifier}: {field} exceeds {TOLERANCE}")
        require(int(regenerated.get("detection_count", 0)) > 0, f"{identifier}: no regenerated detections")

        kernel = AUDIT_ROOT / "kernels" / account / f"wp03-v12-alw-t4-audit-s{seed}-{replica}" / "audit.ipynb"
        payload = kernel.read_bytes()
        require(b"load_state_dict(state[\\\"model\\\"], strict=True)" in payload, f"{identifier}: v12 kernel lacks strict checkpoint load")
        rows.append({
            "id": identifier,
            "manifest_sha256": expected_manifest,
            "official_primary_max_delta": regenerated["official_primary_max_delta"],
            "paper_primary_max_delta": regenerated["paper_primary_max_delta"],
        })

    for seed, manifest_path in SOURCE_MANIFESTS.items():
        expected = next(value[3] for value in REPORTS.values() if value[1] == seed)
        require(sha256(manifest_path) == expected, f"seed {seed}: source manifest no longer matches v12 reports")
        output_name = f"alw_canonical_s{seed}"
        run_dir = next((V8_ROOT / output_name / "runs").iterdir())
        config = load_json(run_dir / "config.json")
        require(config.get("method") == "alw_canonical" and config.get("seed") == seed, f"seed {seed}: v8 config identity mismatch")
        require(config.get("code", {}).get("trainer_sha256") == TRAINER_SHA256, f"seed {seed}: trainer hash mismatch")

    ledger = LEDGER_PATH.read_text(encoding="utf-8")
    require(
        "PL-003 WP03 v12 T4 reproducibility re-adjudication (FROZEN, 2026-08-14)" in ledger,
        "PL-003 owner approval is not frozen",
    )

    return {
        "status": "PASS",
        "scope": "diagnostic_only_no_training_no_test",
        "tolerance": TOLERANCE,
        "reports": rows,
        "promotion_authorized": False,
        "owner_approval_recorded": True,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, sort_keys=True))
