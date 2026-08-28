# Paper A Pre-Run Report: WP03 v9 Independent T4 ALW Audit

Status: `APPROVED_FOR_DIAGNOSTIC_AUDIT` — user authorized execution on
2026-08-10.

## Purpose and Scope

This is not a new training experiment. It diagnoses the two frozen WP03 v8
`alw_canonical` checkpoints that passed artifact integrity but failed the
primary-official endpoint after two independent local CUDA reloads. It asks
whether the discrepancy persists when the same checkpoint is re-inferred on
fresh Tesla T4 workers, the accelerator used by the original kernel.

The audit is validation-only. It does not train, tune, alter NMS or
determinism settings, access final-test material, add a result ledger row, or
promote a scientific claim.

## Immutable Inputs

Each audit account receives a new private input dataset, not a shared kernel
output reference. It contains only the raw original v8 download:

- `best.pt`, `config.json`, `results.json`, `detections_best.json`,
  `metrics.csv`, and `val_gt_paper_primary.json`;
- original `mount_report.json`, `summary.json`, and kernel log;
- `artifact_manifest.json` with SHA-256 for every payload file.

The audit kernel also mounts that account's already-smoked private TinyPerson
data and WP02 code packages. Before inference it asserts: method
`alw_canonical`; requested seed; eight epochs; validation-only result; the
frozen trainer SHA-256
`7c05831cbc544b84926694ecdd85159a9ac85ee557a7dc6894bebcfaed2b5d03`; train
annotation SHA-256
`5bea11d2d6c4f0e524455d7394492eff85991cb6140987573e8890806f9f026b`; and
validation annotation SHA-256
`31d67f94a62d3d9ecbbf825a9dca0a21b22b1a297645dfc34402c59cab50ab27`.

## Execution Assignment

| checkpoint seed | audit account | kernel | replica |
|---:|---|---|---|
| 123 | `hngngnguynvn` | `wp03-v12-alw-t4-audit-s123-r1` | 1 |
| 123 | `quangnhtng` | `wp03-v12-alw-t4-audit-s123-r2` | 2 |
| 2024 | `hngtrngtn` | `wp03-v12-alw-t4-audit-s2024-r1` | 1 |
| 2024 | `luongsythanh` | `wp03-v12-alw-t4-audit-s2024-r2` | 2 |

Each uses `--accelerator NvidiaTeslaT4`. The four accounts have previously
downloaded T4/package-mount health evidence. Credential rotation, dataset
uploads, kernel pushes, status polls, and downloads are serialized.

## Packaging Correction and Current Execution State

The initial v11 replicas are invalid pre-inference attempts: their legacy
`tinyperson-wp01-a1` sources contained only `datasets-metadata.json` on
Kaggle, so `splits/tinyperson_train_sub.json` was absent. They are not results.
Each account now mounts a new immutable `tinyperson-wp03-audit-a1` input with
746 images, `dataset_contract.json`, and the frozen train/validation hashes.
The first downloaded v12 preflight (`hngngnguynvn`) proves 2x Tesla T4, the
data/code/artifact mounts, trainer hash, split hashes, artifact manifest, and
image root. The four v12 audit kernels are pushed; this report must be
superseded only by downloaded per-replica outputs and logs.

## Required Output and Acceptance Boundary

The self-contained notebook first replays saved `detections_best.json` with
both frozen evaluators. A replay delta above `1e-12` fails the audit before a
checkpoint is loaded. It then strict-loads `best.pt`, regenerates validation
detections, and writes `audit_report.json` plus
`regenerated_detections.json`. The report records GPU/Torch/CUDA/cuDNN state,
prediction hashes, per-image differences, full paper-primary deltas, and the
primary TinyPerson official deltas.

`KernelWorkerStatus.COMPLETE` is not acceptance. Each replica requires a
downloaded report and log proving the intended audit ran on a T4. These outputs
are diagnostic evidence only; no WP03 ledger/table/promotion decision changes
until both replicas for a checkpoint are artifact-audited against this report.
