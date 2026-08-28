# Paper A Post-Run Report: WP03 Independent T4 ALW Audit

Status: `DIAGNOSTIC_AUDIT_PASS; NO_LEDGER_PROMOTION` — 2026-08-10.

## Scope

This is the post-run record for the diagnostic-only continuation of
`wp03_v9_t4_audit_prerun.md`. It performs no training, tuning, final-test
access, ledger write, or scientific promotion. It validates only the two
already-downloaded WP03 v8 canonical-ALW checkpoints on fresh Kaggle Tesla T4
workers.

## Evidence Gate

All four v12 kernels reached `COMPLETE`, then their output and log were
downloaded. Every log reports two Tesla T4 devices and the terminal marker
`WP03_V9_T4_AUDIT_OK`. All reports prove the frozen trainer SHA, annotations,
immutable v8 artifact manifest, strict checkpoint load, and saved-detection
replay. Saved replay deltas are exactly zero for every paper-primary and
official-primary endpoint.

## Regenerated-Inference Results

The locked diagnostic tolerance is `5e-4` for primary metrics. Values below
are maximum absolute deltas against the original v8 kernel metrics.

| checkpoint seed | replica | account | max paper-primary delta | max official-primary delta | changed validation images | outcome |
|---:|---:|---|---:|---:|---:|---|
| 123 | r1 | `hngngnguynvn` | `4.7645e-7` | `5.0622e-7` | 1,053 | pass |
| 123 | r2 | `quangnhtng` | `5.9650e-9` | `0` | 1,029 | pass |
| 2024 | r1 | `hngtrngtn` | `1.8609e-8` | `0` | 887 | pass |
| 2024 | r2 | `luongsythanh` | `1.8609e-8` | `0` | 884 | pass |

The regenerated detection hashes differ between workers and from the original
saved list, but the metric deltas remain far below the tolerance. This is
direct evidence that the previously observed local CUDA drift does not
reproduce under independent T4 inference for either frozen checkpoint.

## Interpretation and Boundary

The v8 ALW primary-official mismatch is not reproduced by the four fresh T4
replicas. The evidence supports an environment-sensitive inference difference
rather than corrupt checkpoint, data, evaluator, or saved-detection artifacts.
It does **not** retroactively turn WP03 into an accepted matched comparison:
the user-requested audit is diagnostic only, so the WP03 ledger/table,
promotion state, final-test state, and submission claims remain unchanged.

Raw evidence lives in `.runtime/kaggle/wp03/v9_t4_audit/outputs/v12/`:
each replica has `audit_report.json`, `regenerated_detections.json`, and its
Kaggle log.
