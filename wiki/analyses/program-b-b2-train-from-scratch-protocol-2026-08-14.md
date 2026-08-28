---
title: Program B B2 Authorized Train-from-Scratch Protocol - 2026-08-14
type: protocol
created: 2026-08-14
status: BLOCKED_HARDWARE_CONTRACT_NO_TRAINING
sources:
  - wiki/analyses/program-b-b1-evaluator-integration-gate-2026-08-14.md
  - .runtime/local/program_b/b1_revised_source_20260814/program_b_b1_revised_source_manifest.json
  - .runtime/local/program_b/b1_tiled_20260814/train/tile_manifest.json
  - .runtime/local/program_b/b1_tiled_20260814/validation/tile_manifest.json
---

# Program B B2 Authorized Train-from-Scratch Protocol

## Authorization and boundary

The owner authorized Kaggle-controlled B2 execution on 2026-08-14. This is a
new from-scratch validation program; no historical checkpoint, metric, fair-20
validation score, locked test, or external test asset is eligible as B2 evidence.

## Execution status — 2026-08-17

The evaluator-integrated code snapshot was published as private Kaggle dataset
`ngquangnht/tod-program-b-b2-code-20260814`, version `2`; its downloaded source
manifest SHA-256 is
`9b7f351accf613a29c4f6de53bce2e4845e4a6b885c64e774c0b7ecd3b0491ab`.
The required GPU mount/model-init smoke kernel
`ngquangnht/tod-program-b-b2-mount-smoke-20260814` version `3` terminated
`ERROR` before source copy, model initialization, training, or metric output.
Kaggle assigned a Tesla P100 (`sm_60`) rather than the frozen T4 class; the
current Kaggle PyTorch build supports `sm_70+`, so the hardware assertion
correctly stopped the run. This is a hardware-contract failure, not model or
performance evidence.

The refreshed `b2_baseline_s42` is `BLOCKED_HARDWARE_CONTRACT`; no
evaluator-integrated B2 baseline, PC-MR, PC-MOC, or combined candidate was
submitted. A prior completed kernel with the same baseline label was recovered
and classified `DIAGNOSTIC_ONLY_PRE_B2_PROTOCOL` in
[[Program B B2 Baseline v4 Recovery Audit - 2026-08-17]]: it predates this
source/evaluator contract and cannot serve as a B2 result or teacher.
Locked/external-test access remains none. Retry only with a T4-capable
assignment, or revise the frozen hardware contract through an explicit owner
decision.

## Frozen data/evaluation contract

- train/validation originals: `628/118`, source-group-disjoint;
- train/validation tile manifests: SHA-256
  `6175e8ef8b74a3534c3a8e0227ed40a251a4fe136b902870e6c3112b2604d755` /
  `8b61d70256bbd5196b73b342d20e81471608bba0477227f284f4e01cbe4cc985`;
- geometry: 512 crop, 64 overlap, detector transform 640/800;
- validation: tile detections must reconstruct/NMS to original images before
  the pinned TinyPerson official evaluator;
- selected checkpoint and every reported metric must use this original-image
  official evaluation. Tile-level mAP is diagnostic-only and cannot select or
  rank B2 models.

## Initial matrix

Each arm is a fresh 20-epoch, EMA-enabled, seed-42 run with the same optimizer,
training data, transform, and tile sampler. The baseline runs first; its freshly
produced checkpoint is the only permitted teacher for PC arms.

1. `b2_baseline_s42`: iterative-CBL baseline.
2. `b2_pc_mr_s42`: baseline teacher plus frozen PC-MR-RPN configuration.
3. `b2_pc_moc_s42`: baseline teacher plus frozen PC-MOC-FD configuration.
4. `b2_pc_mr_moc_s42`: baseline teacher plus the tested joint PC-MR/PC-MOC
   configuration.

A subsequent multi-seed matrix is conditional on artifact/reload validation of
this initial matrix; no single-seed outcome makes a performance claim.

## Required artifacts per run

- command/config/source/data hashes and runtime environment;
- epoch checkpoints plus optimizer/EMA state;
- per-epoch original-image official-evaluator rows and selected-checkpoint rule;
- manifest-backed raw tile predictions or deterministic regeneration inputs;
- failure marker on non-success;
- downloaded artifact audit and independent local reload/evaluation.

## Prohibited

- external or locked-test access;
- pretraining or warm-starting from historical B1/fair-20 checkpoints;
- selecting from tile-level metrics;
- changing data, transform, teacher, schedule, or evaluator after inspecting B2
  performance.
