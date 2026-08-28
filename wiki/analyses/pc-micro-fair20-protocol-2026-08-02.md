---
title: PC Micro Fair-20 Protocol - 2026-08-02
type: analysis
created: 2026-08-02
updated: 2026-08-02
sources:
  - runs/pc_moc_fd_seed2718_gate_result.json
  - runs/pc_mr_rpn_r2_seed2718_gate_result.json
tags: [tiny-object, rpn, fpn, distillation, pcgrad, kaggle, fair20]
---

# PC Micro Fair-20 Protocol - 2026-08-02

## Status

**Running.** PC-MOC-FD and PC-MR-RPN independently passed the same frozen
seed-2718 performance gate. They now share one same-source fair-20 baseline.
The locked test is consumed and remains closed.

## Why Both Advance

Independent seed-2718 reload deltas show complementary strengths:

| Candidate | AP | AP50 | AP75 | AR100 | mAP(scale) |
|---|---:|---:|---:|---:|---:|
| PC-MOC-FD | `+0.0063` | `+0.0208` | `+0.0057` | `+0.0194` | `+0.0131` |
| PC-MR-RPN | `+0.0097` | `+0.0327` | `+0.0041` | `+0.0140` | `+0.0335` |

PC-MR is stronger on AP, AP50, and scale mAP; PC-MOC is stronger on AP75 and
AR100. Both improve AP and AP75 on both original-image folds.

## Frozen Schedule

All arms use seed `42`, 20 epochs, EMA, batch size `4`, fixed student transform
`640/800`, copy-paste and tile sampling unchanged, and validation-mAP50
selection of `best.pt`. The source bundle SHA-256 is
`e3c1274ce8ad5917ad7f060389e4a8b56c8e10fa57c71e324b7a20c824258111`.
Candidates use the exact fair20 EMA epoch-5 teacher SHA-256
`90043edfd278a51eef76c8494f4edae8e37127e78fc79dda9eee8071cc29769a`
at `960/1200` only during training.

| Arm | Private Kaggle kernel | Status |
|---|---|---|
| baseline | `ngquangnht/tod-icbl-pcmicro-fair20-s42-20260802` | `RUNNING` |
| PC-MOC-FD | `hngngnguynvn/tod-pcmoc-fd-fair20-s42-20260802` | `RUNNING` |
| PC-MR-RPN | `qnhat1504/tod-pcmr-rpn-fair20-s42-20260802` | `RUNNING` |

## Cloud Smoke Gate

All three smokes ran on exactly two Tesla T4 GPUs. Total losses were
`6.270595`, `6.357537`, and `6.293095` for baseline, PC-MOC, and PC-MR.
Candidate auxiliary losses were finite, both selected `99/131` micro GTs,
PCGrad executed on the intended scope, teacher gradients were zero, and no
teacher state appeared in the student checkpoint.

## Promotion Gate

Each candidate is compared separately with the shared baseline. Promotion
requires exact protocol/source/teacher hashes, 20 metric rows, EMA `best.pt`
selected by validation mAP50, stored-metric/checkpoint agreement, positive
full-validation AP and AP75 deltas, AR100 delta at least `-0.005`, positive AP
on both original-image folds, each fold AP75 delta at least `-0.001`, and no
simultaneous class-aware micro/tiny regression.

The frozen auditors are `.runtime/kaggle/pc_micro_fair20_seed42/audit_pc_moc.py`
and `audit_pc_mr.py`. Terminal artifacts download to
`C:/tmp/tod_pc_micro_fair20_s42`. No locked-test evaluation is authorized.

## Related Pages

- [[PC-MOC-FD Gates - 2026-08-02]]
- [[Post-CR-SC-CBL Mechanism Gates - 2026-08-02]]
- [[PC-MHFD Gates - 2026-08-02]]
- [[RA-TB-CBL Fair-20 Protocol - 2026-08-02]]
- [[Wiki Overview]]
- [[Wiki Log]]
