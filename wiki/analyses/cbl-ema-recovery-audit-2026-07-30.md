---
title: CBL EMA Recovery Audit - 2026-07-30
type: analysis
created: 2026-07-30
updated: 2026-07-30
sources: [kaggle:tod-cbl-ema8-20260730, common/model.py, scripts/train_frcnn_metric.py, runs/kaggle_cbl_ema8_best_ap75_valid_reload.json]
tags: [cbl, ema, kaggle, checkpoint, validation, ap75]
---

# CBL EMA Recovery Audit - 2026-07-30

## Question

Can the non-reloadable EMA peak from the 20-epoch CBL run be recovered as an
exact checkpoint after fixing the checkpoint contract?

## Run Contract

- private kernel: `quangnhtng/tod-cbl-ema8-20260730`;
- source branch: `cbl-ema-checkpoint-fix-20260730`;
- pinned commit: `40db904a795d0abd2847ae66646d046390c88315`;
- seed 42, eight epochs, CBL RoI localization, SA-ALW assignment;
- EMA, copy-paste, and tiny-tile oversampling enabled;
- best checkpoints save the exact evaluated EMA state and declare
  `model_source=ema`; `last.pt` keeps raw weights plus EMA resume state.

## Artifact Audit

The downloaded output contains all eight metric rows, the kernel log, and four
checkpoints: `best.pt`, `best_ap75.pt`, `best_coco_ap.pt`, and `last.pt`.
The three best checkpoints all select EMA epoch 5 and their stored validation
metrics are reloadable.

| Epoch | COCO AP | AP50 | AP75 | AR100 | weighted class-aware AP | micro class-aware AP |
|---:|---:|---:|---:|---:|---:|---:|
| 5, stored EMA | 0.1408 | 0.3891 | 0.0664 | 0.2946 | 0.5272 | 0.3703 |
| 5, independent reload | **0.1409** | **0.3891** | **0.0665** | **0.2947** | **0.5270** | **0.3697** |
| 8, stored EMA | 0.1284 | 0.3652 | 0.0536 | 0.2770 | 0.5019 | 0.3783 |

The reload difference is at most `0.0002`, consistent with evaluator ordering
noise. The checkpoint contract is therefore fixed in practice, not only in
metadata.

## Comparison

The reloadable EMA epoch-5 checkpoint improves over the legacy raw epoch-5 CBL
checkpoint on validation:

| Checkpoint | AP | AP50 | AP75 | AR100 | weighted class-aware AP |
|---|---:|---:|---:|---:|---:|
| Legacy raw epoch 5 | 0.1277 | 0.3659 | 0.0554 | 0.2768 | 0.5182 |
| Recovery EMA epoch 5 | **0.1409** | **0.3891** | **0.0665** | **0.2947** | **0.5270** |

This is the current validation leader among audited reloadable CBL
checkpoints. It is slightly below the original run's non-reloadable stored EMA
peak (`0.1440/0.0677` AP/AP75), which is expected run-to-run CUDA
nondeterminism rather than a checkpoint mismatch.

## Residual Error Diagnosis

Caching raw tile predictions at score threshold `0.001` shows that the EMA
checkpoint improves both score ordering and candidate localization:

| Checkpoint | Pearson score-IoU | Spearman score-IoU | IoU50 GT coverage | IoU75 GT coverage |
|---|---:|---:|---:|---:|
| Local CBL epoch 2 | 0.6181 | 0.5225 | 0.7312 | 0.2115 |
| Recovery EMA epoch 5 | **0.6528** | **0.5349** | **0.7565** | **0.2369** |

The EMA mean score remains calibrated at `0.0829`, unlike the failed sampled
Rank & Sort checkpoint (`0.4483`). Classification ranking is no longer the
primary bottleneck. The next research direction should prioritize iterative
box refinement and tile-boundary-aware localization while preserving the
current classifier and CBL score path.

## Decision

Promote recovery `best_ap75.pt` as the current reloadable CBL validation
candidate. Do not evaluate it on the locked test set now: the CBL family has
already consumed its single frozen test gate with the legacy raw epoch-5
checkpoint. A second look would contaminate model selection.

Use this checkpoint for validation-only calibration, ensembling, or future
research. Any future locked-test use requires a predeclared final-candidate
decision that is independent of additional test feedback.

## Artifacts

- `.runtime/kaggle/cbl_ema8/artifact_audit.json`
- `.runtime/kaggle/cbl_ema8/output/tod_output/runs/sa_alw_full__cbl__la_loss__seed42__cbl_ema8/best_ap75.pt`
- `runs/kaggle_cbl_ema8_best_ap75_valid_reload.json`
- `runs/quality_diagnosis_valid_cbl_ema8_ep5.json`

## Related Pages

- [[Confidence-Driven Localization Local Gate - 2026-07-30]]
- [[CBL Rank and Sort Local Gate - 2026-07-30]]
- [[Test-Set Evaluation — Phase 2 Metrics]]
