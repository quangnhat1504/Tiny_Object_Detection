---
title: CR-SC-CBL Multi-Seed Fair-20 Protocol - 2026-08-02
type: analysis
created: 2026-08-02
updated: 2026-08-02
sources:
  - wiki/analyses/coordinate-reliable-sc-cbl-plan-2026-08-01.md
  - wiki/analyses/iterative-cbl-fair20-locked-test-protocol-2026-08-01.md
tags: [cbl, distillation, kaggle, multi-seed, fair-comparison]
---

# CR-SC-CBL Multi-Seed Fair-20 Protocol - 2026-08-02

## Status

The three-seed validation matrix is frozen and running. Four new private
fair-20 kernels are `RUNNING`; the seed-123 CR-SC-CBL reference launched on
2026-08-01 is also `RUNNING`. A completed smoke is not a performance result,
and none of these runs is a paper checkpoint until artifacts and independent
validation reloads pass.

The locked-test budget remains consumed and closed. This matrix uses only the
training and validation data.

## Fair Matrix

Each pair uses 20 fresh epochs, the same SGD/warmup/cosine schedule, batch size
4, EMA, fixed `640/800` student transform, validation-mAP50 selection of
`best.pt`, and the same trainable/inference iterative-CBL settings. The only
method difference is CR-SC-CBL at frozen loss weight `0.25`, temperature `2`,
teacher transform `960/1200`, and coordinate-reliable weighting.

| Seed | Iterative-CBL baseline | CR-SC-CBL candidate |
|---:|---|---|
| 42 | `quangnhtng/tod-cbl-itrain-fair20-20260801` (complete reference) | `hngngnguynvn/tod-cr-sc-cbl-fair20-s42-20260802` (`RUNNING`) |
| 123 | `ngquangnht/tod-icbl-fair20-s123-20260802` (`RUNNING`) | `quangnhtng/tod-cr-sc-cbl-fair20-20260801` (`RUNNING`) |
| 2024 | `amongus1504/tod-icbl-fair20-s2024-20260802` (`RUNNING`) | `qnhat1504/tod-cr-sc-cbl-fair20-s2024-20260802` (`RUNNING`) |

All newly generated notebooks are self-contained. The locked source bundle
SHA-256 is
`6df703015677a50860a5a9c3c4ae3fad5f5f4c281b8a03668781bf7420fb5c5d`.
The frozen teacher checkpoint is exactly `330630397` bytes with SHA-256
`90043edfd278a51eef76c8494f4edae8e37127e78fc79dda9eee8071cc29769a`.
Two private teacher datasets expose that exact artifact to candidate accounts:

- `hngngnguynvn/tod-fair20-teacher-20260802`;
- `qnhat1504/tod-fair20-teacher-20260802`.

## Account and Smoke Audit

Seven configured Kaggle credentials authenticated successfully and could read
the public training dataset. Five accounts currently host the long fair-20
runs; the remaining two are reserved for failures or later promoted methods.
No credential material is recorded in the wiki.

All four new kernels first completed a private two-T4 smoke:

| Run | Total loss | Distill loss | Teacher contract |
|---|---:|---:|---|
| baseline seed 123 | 6.344304 | n/a | n/a |
| CR-SC-CBL seed 42 | 6.547282 | 0.274967 | exact hash, zero teacher gradients |
| baseline seed 2024 | 6.146940 | n/a | n/a |
| CR-SC-CBL seed 2024 | 6.401575 | 0.252296 | exact hash, zero teacher gradients |

Every smoke reported two Tesla T4 GPUs. Candidate smokes also confirmed that
teacher parameters are not present in the student state dict.

## Artifact Gate

For every long run:

1. wait for a terminal Kaggle state;
2. download outputs rather than trusting notebook status;
3. reject any run with `failure.json`, missing `metrics.json`, fewer than 20
   metric rows, missing checkpoints, or a protocol/hash mismatch;
4. independently reload the validation-selected `best.pt`;
5. compare paired per-seed AP, AP50, AP75, AR100, class-aware scale AP, and
   both original-image folds; and
6. report mean, standard deviation, and the three paired deltas before any
   paper promotion decision.

The serialized manager and state live under
`.runtime/kaggle/cr_sc_cbl_multiseed/`. One serial credential-rotation monitor
polls the four new kernels every five minutes and downloads outputs only after
terminal completion. Do not start a second concurrent Kaggle poller because
the CLI credential override is process-global on this machine.

## Claim Boundary

Allowed now: the protocol is frozen, all four new technical smokes passed, and
five long validation kernels are running across five accounts.

Not allowed now: multi-seed improvement, statistical significance, test-set
improvement, or a new paper checkpoint.

## Related Pages

- [[Coordinate-Reliable SC-CBL Plan - 2026-08-01]]
- [[Post-CR-SC-CBL Mechanism Gates - 2026-08-02]]
- [[Iterative CBL Fair-20 Locked-Test Protocol - 2026-08-01]]
- [[Wiki Overview]]
- [[Wiki Log]]
