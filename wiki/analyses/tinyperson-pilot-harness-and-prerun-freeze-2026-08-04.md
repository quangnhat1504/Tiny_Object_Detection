---
title: TinyPerson Pilot Harness and Pre-Run Freeze - 2026-08-04
type: analysis
created: 2026-08-04
updated: 2026-08-04
sources: [paper_a/tools/train_tinyperson_pilot.py, paper_a/experiment_reports/wp01_pilot_prerun.md, paper_a/protocol_ledger.md]
tags: [paper-a, sa-alw, pilot, harness, kaggle]
---

# TinyPerson Pilot Harness and Pre-Run Freeze - 2026-08-04

Question: with PL-001 frozen, can the WP01 TinyPerson G3 pilot get a
canonical training harness, pass local smokes, and reach `READY_FOR_PUSH`?

## What was decided

- **PL-001 frozen** (user-approved): deterministic video/source-disjoint 20%
  validation split. Builder checked in at
  `paper_a/tools/build_tinyperson_validation_split.py`; artifacts hashed:
  `tinyperson_train_sub.json` (6,215 crops / 27,711 positives, sha256
  `5bea11d2...760f9f026b`) and `tinyperson_val.json` (2,041 crops / 4,719
  positives, sha256 `31d67f94...50ab27`). Validation is selection-only; no
  tuning against it, no test access.
- **Harness**: new canonical pilot trainer
  `paper_a/tools/train_tinyperson_pilot.py` (sha256 `38a89023...634ec9f`).
  The legacy SOD harness (`scripts/train_frcnn_metric.py`) was assessed and
  rejected for the pilot: it is coupled to the SOD tile dataset, copy-paste,
  and weighted sampling. The new trainer reuses `common/` primitives
  (`build_model`, `configure_metric`, `train_one_epoch`, `WarmupCosineLR`)
  with a clean Paper A data path:
  - `TinyPersonOriginalDataset` on the PL-001 split files;
  - seeded horizontal flip (p=0.5) as the only augmentation;
  - seeded per-epoch data order identical across methods;
  - `num_classes=1` (binary task-all);
  - checkpoint selector = validation `paper_primary_coco` AP (standard
    COCO eval, ignored/uncertain GT as `iscrowd`), final metrics from an
    independent strict reload;
  - second labeled family `benchmark_official` via the pinned TinyPerson
    evaluator (AP25/50/75, tiny bins);
  - reduced budget frozen at 8 epochs, batch 4, SGD 0.005 with 2-epoch
    warmup + cosine.

## Method map (frozen six)

| Pilot method | Build |
|---|---|
| standard | `metric_fn=None`, placement `everywhere`, Smooth-L1 box loss |
| igwd | verified predecessor, placement `la_loss`, metric box loss |
| alw_canonical | `configure_metric("alw_canonical", beta=8.0)` |
| sa_alw beta-only / pos-only / full | canonical SA metrics with TinyPerson P10/P90 bounds (7.4328/44.8468), beta 8→10, w 1→1.5, linear |

## Smoke evidence

All six method paths ran a 1-epoch / 8-train / 4-val CUDA smoke: losses
finite and decreasing, checkpoint save + strict reload, both evaluator
families emitted. A perfect-detection fixture through the trainer's own GT
builder + evaluators returns AP = 1.0 (`paper_primary_coco`) and
AP25/50/75 = 1.0 (`benchmark_official`) on a 64-crop val subset
(`.runtime/pilot_eval_path_fixture/report.json`). The TinyPerson
corner-task annotation carries no ignore/uncertain flags (erased dataset),
so the `iscrowd` routing is defensive. Smoke checkpoints deleted; no local
pilot numbers exist.

## Governance updates

- `endpoint_protocol.md`: `D1_SCALE_BOUNDS_FITTED` (TinyPerson side).
- `pilot_decision_protocol.md`: `TINYPERSON_SIDE_READY`.
- `assignment_board.csv` WP01: owner `Qoder-Leader`, account `ngquangnht`
  (first of the verified 8-account pool; both pilot shards on the same
  account per protocol), status `READY_FOR_PUSH`.
- Pre-run report: `paper_a/experiment_reports/wp01_pilot_prerun.md`,
  including T4 compute estimate (~20-25 GPU-hours/method) and the
  determinism disclosure (seeded RNGs, cuDNN not bit-deterministic).

## Open

- Package + private Kaggle dataset upload (A1 material only) and record its
  sha256; then push six self-contained T4 kernels; post-run artifact audit.
- AI-TOD-v2 images still pending (blocks WP06, not WP01).
