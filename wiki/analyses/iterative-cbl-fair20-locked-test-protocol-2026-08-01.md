---
title: Iterative CBL Fair-20 Locked-Test Protocol - 2026-08-01
type: analysis
created: 2026-08-01
updated: 2026-08-01
sources: [common/config.py, scripts/train_frcnn_metric.py, scripts/test_coco_eval_single.py]
tags: [cbl, sa-alw, fair-comparison, preregistration, locked-test]
---

# Iterative CBL Fair-20 Locked-Test Protocol - 2026-08-01

## Purpose

Compare the trainable iterative-CBL candidate with historical SA-ALW without
using the locked test for checkpoint selection. The candidate retains SA-ALW
assignment and changes RoI localization to CBL with one trainable refinement
pass.

## Frozen Training Protocol

| Setting | Value |
|---|---|
| Source | `cbl-iterative-depth-20260731` at `80e934aaa7555733d795a8adbe70c19027e67735` |
| Data | unchanged train/valid/test split and 512-pixel tiling |
| Seed | 42 |
| Epoch budget | 20, no early stop |
| Optimizer/schedule | SGD and the shared two-epoch warmup plus 20-epoch cosine schedule |
| Assignment metric | `sa_alw_full`, placement `la_loss` |
| RoI localization | CBL, no metric-loss warmup |
| Iterative training | weight `0.5`, one pass |
| Iterative inference | one pass, blend `1.0`, score threshold `0.30` |
| EMA | enabled, decay `0.9998` |
| Augmentation | copy-paste enabled; tiny-tile oversampling `2.0` |

This is a fresh 20-epoch run. The earlier eight-epoch checkpoint must not be
resumed because changing the cosine horizon after epoch 8 would not reproduce a
20-epoch schedule.

## Frozen Selection Rule

Use `best.pt`, selected only by validation `mAP_50`, matching the historical
SA-ALW checkpoint-selection rule. Training always completes all 20 epochs.
AP, AP75, and AR100 are reported but cannot change the selected checkpoint.

After download, reload `best.pt` independently on the full validation split.
The reload must use the checkpoint's stored model configuration and must match
the stored validation metrics within evaluator rounding tolerance.

## Locked-Test Rule

If the artifact and validation reload audits pass, evaluate exactly one frozen
checkpoint on the 65-image locked test:

```text
best.pt + stored one-step iterative inference configuration
```

Do not evaluate `best_ap75.pt`, `best_coco_ap.pt`, `last.pt`, TTA variants, or
alternative refinement profiles on test. Compare the single result with the
historical SA-ALW seed-42 artifact and retain all gains and regressions in the
report.

## Status

- Protocol frozen before the fair-20 run and locked-test evaluation.
- Private Kaggle kernel: `quangnhtng/tod-cbl-itrain-fair20-20260801`, version 1.
- Source and hardware audit passed: exact commit
  `80e934aaa7555733d795a8adbe70c19027e67735`, PyTorch `2.10.0+cu128`, and
  two Tesla T4 devices.
- Kaggle status `COMPLETE`; all 20 metric rows, four checkpoints, protocol,
  and kernel log downloaded successfully.
- Local manager: `.runtime/kaggle/cbl_iterative_train_fair20/manage_kernel.py`.
- Independent validation reload passed and the one allowed locked-test
  evaluation is complete. Locked-test budget consumed: `1/1`.

## Training Result

Training completed all 20 epochs in about 6 hours 47 minutes. Metrics peak
early and then decline monotonically under the shared 20-epoch cosine horizon:

| Selection view | Epoch | mAP50 | COCO AP | AP75 | AR100 |
|---|---:|---:|---:|---:|---:|
| Frozen `best.pt` by mAP50 | **5** | **0.3999** | **0.1456** | 0.0711 | **0.2959** |
| Best AP75, ineligible for test | 4 | 0.3936 | 0.1444 | **0.0743** | 0.2953 |
| Final epoch | 20 | 0.3230 | 0.1064 | 0.0407 | 0.2371 |

The full budget therefore confirms that epoch 5 is a genuine validation peak;
later epochs do not recover. `best.pt` contains exact EMA epoch-5 weights and
the frozen iterative-CBL configuration.

## Independent Validation Reload

The full 1,764-tile reload reproduced the stored checkpoint metrics within
`0.0000-0.0002` evaluator rounding:

| AP | AP50 | AP75 | AR100 | mAP(scale) |
|---:|---:|---:|---:|---:|
| 0.1456 | 0.3969 | 0.0711 | 0.2959 | 0.6162 |

This passed the preregistered gate before any locked-test evaluation.

## Locked-Test Result

Exactly one evaluation was run on the 65-image, 826-tile locked test using
`best.pt` and its stored one-step inference configuration.

| Metric | Historical SA-ALW | Fair-20 iterative CBL | Absolute delta | Relative delta |
|---|---:|---:|---:|---:|
| COCO AP | 0.0975 | **0.1158** | +0.0183 | +18.77% |
| AP50 | 0.3058 | **0.3326** | +0.0268 | +8.76% |
| AP75 | 0.0344 | **0.0533** | +0.0189 | +54.94% |
| AR100 | 0.2509 | **0.2657** | +0.0148 | +5.90% |
| mAP(scale) | 0.6014 | **0.6130** | +0.0116 | +1.93% |
| custom mAP50 | 0.3110 | **0.3375** | +0.0265 | +8.51% |

The fair-20 candidate improves every primary locked-test metric, including the
strict-localization AP75 objective. It is the new project test leader among the
recorded single-checkpoint results. The result supports the combined method:
SA-ALW assignment plus CBL localization and one trainable refinement pass.

No other fair-20 checkpoint or inference variant may now be evaluated on this
locked test.

## Result Artifacts

- `runs/kaggle_cbl_iterative_train_fair20_best_map50_valid_reload.json`
- `runs/kaggle_cbl_iterative_train_fair20_best_map50_locked_test.json`
- `.runtime/kaggle/cbl_iterative_train_fair20/output/tod_output/protocol.json`
- `.runtime/kaggle/cbl_iterative_train_fair20/output/tod_output/runs/sa_alw_full__cbl__irtw0.5ir1s0.3__la_loss__seed42__cbl_iterative_train_fair20/metrics.csv`

## Resume Commands

From the project root:

```powershell
.venv-cuda\Scripts\python.exe .runtime\kaggle\cbl_iterative_train_fair20\manage_kernel.py status
.venv-cuda\Scripts\python.exe .runtime\kaggle\cbl_iterative_train_fair20\manage_kernel.py download
```

The run is complete and downloaded. These commands remain useful for status
re-verification or artifact recovery; they must not trigger another test
evaluation.
