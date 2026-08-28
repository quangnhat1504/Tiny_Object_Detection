# WP03 A2 T4 Re-adjudication Amendment — Proposal

**Ledger entry:** `PL-003`  
**State:** `FROZEN — owner accepted 2026-08-14`  
**Decision:** v12 T4 diagnostics are accepted as reproducibility evidence for
the two immutable WP03-v8 canonical-ALW checkpoints.

## Proposed decision

Accept the four downloaded v12 reports as the platform reproducibility
evidence for only these existing validation-only artifacts:

| v8 method | Seed | v12 replicas | immutable manifest SHA-256 |
|---|---:|---|---|
| `alw_canonical` | 123 | `hngngnguynvn/r1`, `quangnhtng/r2` | `b1fe988f58d865c885498c4dea13ec8308847a4fdd9422a9d1176ca394b28cd2` |
| `alw_canonical` | 2024 | `hngtrngtn/r1`, `luongsythanh/r2` | `806bffbe3c2ebea6a4eeb0aa23f8ea299d5d1702e4297c72409b06d7caacd122` |

The reports are diagnostic-only, use two Tesla T4 GPUs each, replay saved
detections exactly, strictly load the frozen checkpoint, and regenerate every
primary endpoint within `5e-4` (observed maximum `5.0622038e-7`).

## Invariants retained

- The original local ALW reload failures remain recorded; this amendment does
  not rewrite them or loosen the `5e-4` tolerance.
- Data, PL-001 split, trainer SHA-256
  `7c05831cbc544b84926694ecdd85159a9ac85ee557a7dc6894bebcfaed2b5d03`,
  checkpoints, predictions, and evaluator payloads remain immutable.
- v12 did not retrain a method, improve a metric, access final test data, or
  establish a scientific effect.
- Approval only makes the two ALW v8 rows eligible for the validation ledger
  alongside the already-passing SA-ALW rows. It does not complete the matrix,
  permit final-test use, or authorize a Paper A claim.

## Consequences

The owner accepted this amendment on 2026-08-14. Prepare exactly two validation-only, WP02-hash seed-42 shards
(`alw_canonical` and canonical full SA-ALW) under a separate approved pre-run
report. No GPU work is authorized by this amendment alone.

## Verification

```powershell
.\.venv-cuda\Scripts\python.exe paper_a\tools\validate_wp03_v12_reproducibility.py
```

This validator is read-only and must pass again immediately before an owner
freezes `PL-003`.
