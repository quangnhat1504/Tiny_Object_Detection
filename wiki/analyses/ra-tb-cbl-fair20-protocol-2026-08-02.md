---
title: RA-TB-CBL Fair-20 Protocol - 2026-08-02
type: analysis
created: 2026-08-02
updated: 2026-08-02
sources:
  - wiki/analyses/post-cr-sc-cbl-mechanism-gates-2026-08-02.md
  - wiki/analyses/iterative-cbl-fair20-locked-test-protocol-2026-08-01.md
tags: [cbl, distillation, kaggle, fair-comparison, validation-only]
---

# RA-TB-CBL Fair-20 Protocol - 2026-08-02

## Status

The paired seed-42 validation run is frozen and `RUNNING` on Kaggle. Both
private preflight smokes completed on exactly two Tesla T4 GPUs and passed the
source, teacher, gradient-isolation, and serialization contracts. This is not
a performance result; promotion requires downloaded artifacts and independent
validation reloads.

The locked-test budget remains consumed and closed. This experiment uses only
training and validation data.

## Frozen Pair

Both jobs use 20 fresh epochs, seed `42`, batch size `4`, EMA, the same
SGD/warmup/cosine schedule, fixed `640/800` student transform, and `best.pt`
selection by the maximum validation mAP50 row. The identical self-contained
source bundle has SHA-256
`6cdd1d0fcde403386a19fc10d4efc5af3e11e1e1c499f863a0ed30b86d879966`.

| Arm | Private Kaggle kernel | Method-only difference |
|---|---|---|
| baseline | `thyngluthy/tod-icbl-fair20-s42-r2-20260802` | iterative CBL only |
| candidate | `hienquang06/tod-ra-tb-cbl-fair20-s42-20260802` | RA-TB auxiliary with frozen `960/1200` teacher |

RA-TB keeps post-refinement alignment and uses the high-resolution teacher
only to select coordinates where it is reliably better. Selected coordinates
optimize the exact ground-truth two-bin CBL target, not teacher-logit KL. The
frozen teacher checkpoint SHA-256 is
`90043edfd278a51eef76c8494f4edae8e37127e78fc79dda9eee8071cc29769a`.

## Smoke Evidence

| Arm | Total loss | RA-TB loss | Contract |
|---|---:|---:|---|
| baseline | `6.271968` | n/a | two T4 GPUs, source locked |
| candidate | `6.736825` | `0.466231` | exact teacher hash, zero teacher gradients, no teacher state duplication |

The smokes use the same package as the long jobs. A completed smoke proves
only executable integrity, not detector quality.

## Artifact And Decision Gate

The serial poller and state live under
`.runtime/kaggle/ra_tb_cbl_fair20_seed42/`; terminal outputs download to
`C:/tmp/tod_rtb_fair20_s42`. The prepared auditor is
`.runtime/kaggle/ra_tb_cbl_fair20_seed42/audit_fair20.py`.

Before any claim:

1. download both terminal outputs and inspect `failure.json`;
2. require exactly 20 metric rows, checkpoints, config, source hash, seed,
   schedule, EMA, and teacher contracts;
3. require `best.pt` to match the row with maximum validation mAP50;
4. independently reload both checkpoints on full validation and both frozen
   original-image folds; and
5. compare AP, AP50, AP75, AR100, mAP(scale), and class-aware scale AP.

Paper-checkpoint promotion requires the candidate to improve full-validation
AP and AP75, preserve AR100 within `-0.005`, improve AP on both folds, keep
each fold's AP75 delta at least `-0.001`, and avoid simultaneous regression
on class-aware micro and tiny AP. A failed artifact contract is a failed run,
not permission to infer a result from notebook status.

## Claim Boundary

Allowed now: the seed-31415 short gate passed all preregistered conditions,
the equal-schedule seed-42 fair-20 pair is running, and both technical smokes
passed.

Not allowed now: a fair-20 performance gain, a new paper checkpoint, a
locked-test comparison, or tuning from test evidence.

## Related Pages

- [[Post-CR-SC-CBL Mechanism Gates - 2026-08-02]]
- [[RA-TB plus PC-MHFD Combination Gates - 2026-08-02]]
- [[Iterative CBL Fair-20 Locked-Test Protocol - 2026-08-01]]
- [[Wiki Overview]]
- [[Wiki Log]]
