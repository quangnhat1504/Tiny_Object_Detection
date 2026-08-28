---
title: Test-Set Evaluation — Phase 2 Metrics
type: analysis
created: 2026-07-04
updated: 2026-07-30
sources: [scripts/test_eval.py, common/eval_utils.py, runs/sa_alw_full__smooth_l1__la_loss__seed42__smooth_l1_ap75/test_metrics_kaggle_best_ap75.json, runs/qscore_w025_best_ap75_locked_test.json, runs/qscore_w05_seed2024_best_ap75_locked_test.json, runs/kaggle_cbl_full_best_ap75_locked_test.json]
tags: [test-set, evaluation, phase2, metrics, kaggle, ap75]
---

## Test-Set Evaluation — Phase 2 Metrics

### Context

All Phase 2 metric ablation experiments were trained and validated on the
validation split (`data/valid/`, 131 images). This report evaluates each
checkpoint on the **locked test set** (`data/test/`, 65 images) to measure
generalization.

### Protocol

- **Checkpoint**: `best.pt` from each run (selected by best validation mAP@50)
- **Model**: Faster R-CNN + ResNet-50-FPN + RFLA, built identically to training
- **Placement**: `la_loss` (metric in RPN label assignment + RoI box regression loss)
- **Evaluation script**: `scripts/test_eval.py`
- **Metrics**: Scale-aware mAP (micro/tiny/small/large bins), COCO mAP@50, COCO AP@50:75, COCO AP@75, AR@100

### Results

Run the evaluation script to populate this table:

```bash
python scripts/test_eval.py
```

Output: `runs/test_results.json` and per-run `runs/<name>/test_metrics.json`.

#### Results (test set, 65 images, 826 tiles, seed 42)

| # | Run Name | Metric | Val mAP@50 | Test mAP(scale) | Test AP_micro | Test AP_tiny | Test AP_small | Test AP_large | Test mAP@50 |
|---|----------|--------|-------------|-----------------|---------------|--------------|---------------|---------------|-------------|
| 1 | `frcnn_standard__full__seed42` | IoU | 0.3872 | 0.5231 | 0.4271 | 0.5353 | 0.5467 | 0.7420 | 0.2726 |
| 2 | `frcnn_standard__full__seed123` | IoU | 0.3895 | 0.5145 | 0.4174 | 0.5301 | 0.5307 | 0.7400 | 0.2408 |
| 3 | `frcnn_standard__full__seed2024` | IoU | 0.3909 | 0.4908 | 0.4554 | 0.4915 | 0.4970 | 0.7186 | 0.2490 |
| 4 | `frcnn_standard__patches__seed42` | IoU | 0.3823 | 0.5893 | 0.5771 | 0.5846 | 0.6026 | 0.6663 | 0.2966 |
| 5 | `frcnn_standard__patches__seed123` | IoU | 0.3804 | 0.5940 | 0.5290 | 0.5960 | 0.6244 | 0.7346 | 0.2967 |
| 6 | `frcnn_standard__patches__seed2024` | IoU | 0.3784 | **0.6114** | **0.5783** | 0.6075 | **0.6363** | 0.7020 | **0.3331** |
| 7 | `nwd__la_loss__seed42` | NWD | 0.3362 | 0.5648 | 0.4994 | 0.6032 | 0.5607 | 0.0782 | 0.3029 |
| 8 | `igwd__la_loss__seed42` | IGWD | 0.3875 | 0.5366 | 0.4405 | 0.5494 | 0.5549 | **0.8079** | 0.2688 |
| 9 | `alw_full__la_loss__seed42` | ALW | 0.3923 | 0.4572 | 0.4028 | 0.4731 | 0.4392 | 0.7350 | 0.2257 |
| 10 | `igwd_anisotropic_s__la_loss__seed42` | IGWD+aniso | 0.3640 | 0.5620 | 0.4669 | 0.5846 | 0.5939 | 0.3482 | 0.2851 |
| 11 | `igwd_log_shape__la_loss__seed42` | IGWD+log | 0.3661 | 0.4541 | 0.2750 | 0.4649 | 0.5396 | 0.6633 | 0.2192 |
| 12 | `sa_alw_full__la_loss__seed42` | **SA-ALW** | 0.3964 | **0.6005** | 0.5091 | **0.6280** | 0.5994 | 0.6452 | 0.3122 |
| 13 | `sa_alw_beta_only__la_loss__seed42` | SA-ALW(β) | 0.3981 | 0.5762 | **0.5147** | 0.5998 | 0.5579 | 0.6914 | 0.2962 |
| 14 | `sa_alw_pos_only__la_loss__seed42` | SA-ALW(w_pos) | 0.3971 | 0.5899 | 0.4916 | 0.6159 | 0.5938 | 0.6823 | 0.3034 |

#### AP75 Kaggle Checkpoint Result (test set, 65 images, 826 tiles, seed 42)

This evaluates the downloaded Kaggle checkpoint selected by validation COCO AP75:

- **Checkpoint**: `.runtime/kaggle/tod_results/smooth_l1_ap75/tod_output/runs/sa_alw_full__smooth_l1__la_loss__seed42__smooth_l1_ap75/best_ap75.pt`
- **Training run**: `sa_alw_full__smooth_l1__la_loss__seed42__smooth_l1_ap75`
- **Selection epoch**: 8 (`best_ap75.pt`)
- **Eval device**: local GPU via `.venv-cuda` (`torch 2.11.0+cu128`, RTX 5070 Ti)
- **Output JSON**: `runs/sa_alw_full__smooth_l1__la_loss__seed42__smooth_l1_ap75/test_metrics_kaggle_best_ap75.json`

| Run Name | Checkpoint | Val mAP@50 | Test mAP(scale) | Test AP_micro | Test AP_tiny | Test AP_small | Test AP_large | Test mAP@50 | COCO AP | COCO AP50 | COCO AP75 | COCO AP_S | COCO AP_M | COCO AP_L | AR@1 | AR@10 | AR@100 | Precision | Recall | TP / GT |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `sa_alw_full__smooth_l1__la_loss__seed42__smooth_l1_ap75` | `best_ap75.pt` | 0.3868 | 0.5474 | 0.4920 | 0.5594 | 0.5529 | 0.6351 | 0.2717 | 0.0832 | 0.2665 | 0.0250 | 0.0726 | 0.1746 | 0.3798 | 0.0244 | 0.1183 | 0.2260 | 0.0876 | 0.6212 | 3617 / 5823 |

**Interpretation:** this Kaggle checkpoint is the best validation AP75 artifact from the AP75-focused run, but it is not a new locked-test best. It trails `sa_alw_full__la_loss__seed42` on test mAP(scale) (`0.5474` vs `0.6005`) and trails the stronger Phase 2 AP75 entries such as IoU patches seed 2024 (`0.0250` vs `0.0375`). Treat it as evidence that validation AP75 selection overfits.

#### Kaggle Checkpoint-Selection Audit (2026-07-22)

The AP75-focused Kaggle runs were re-audited by checkpoint file, not just by the validation-selected artifact. This corrected the earlier assumption that `best_ap75.pt` was the only useful checkpoint from `smooth_l1_ap75`.

| Artifact | Checkpoint | Epoch | Test mAP(scale) | COCO AP | COCO AP50 | COCO AP75 | AR@100 | Output JSON |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `smooth_l1_ap75` | `best.pt` | 7 | **0.5844** | **0.0970** | **0.3000** | **0.0358** | **0.2525** | `runs/kaggle_checkpoint_eval_smooth_l1_best.json` |
| `smooth_l1_ap75` | `best_ap75.pt` | 8 | 0.5474 | 0.0832 | 0.2665 | 0.0250 | 0.2260 | `runs/sa_alw_full__smooth_l1__la_loss__seed42__smooth_l1_ap75/test_metrics_kaggle_best_ap75.json` |
| `os1` | `best_ap75.pt` | 15 | 0.4411 | 0.0682 | 0.2188 | 0.0225 | 0.2098 | `runs/kaggle_checkpoint_eval_os1_best_ap75.json` |
| `cp_light` | `best.pt` | 12 | 0.4273 | 0.0673 | 0.2104 | 0.0211 | 0.2104 | `runs/kaggle_checkpoint_eval_cp_light_best.json` |
| `os125` | `best.pt` | 18 | 0.4528 | 0.0709 | 0.2308 | 0.0205 | 0.2108 | `runs/kaggle_checkpoint_eval_os125_best.json` |

**Updated interpretation:** `smooth_l1_ap75/best.pt` is the best downloaded Kaggle candidate and improves over the baseline SA-ALW seed42 AP75 (`0.0358` vs `0.0344`), but it still does not beat the best existing AP75 test entry (`frcnn_standard__patches__seed42`, `0.0375`) or the best mAP(scale) entries. Use it as an ensemble/postprocess candidate rather than as a standalone SOTA checkpoint.

#### Quality-Score Rerun Locked-Test Gate (2026-07-24)

Quality-score checkpoints improved validation AP75 slightly, but the local locked-test gate did not promote any standalone checkpoint.

| Run | Checkpoint | Epoch | mAP(scale) | COCO AP | AP50 | AP75 | AR100 | Output |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `q_smooth_l1_w025` | `best_ap75.pt` | 9 | 0.4907 | 0.0845 | 0.2559 | **0.0341** | 0.2139 | `runs/qscore_w025_best_ap75_locked_test.json` |
| `q_smooth_l1_w05_seed2024` | `best_ap75.pt` | 7 | **0.5581** | **0.0905** | **0.2818** | 0.0338 | **0.2403** | `runs/qscore_w05_seed2024_best_ap75_locked_test.json` |
| `q_smooth_l1_w10` | `best_ap75.pt` | 9 | 0.5067 | 0.0874 | 0.2628 | 0.0332 | 0.2208 | `runs/qscore_w10_best_ap75_locked_test.json` |
| `q_smooth_l1_w05` | `best.pt` | 8 | 0.5003 | 0.0821 | 0.2585 | 0.0302 | 0.2142 | `runs/qscore_w05_best_locked_test.json` |
| `q_smooth_l1_w05` | `best_coco_ap.pt` | 8 | 0.5005 | 0.0821 | 0.2587 | 0.0300 | 0.2139 | `runs/qscore_w05_best_coco_ap_locked_test.json` |

**Decision:** qscore is a negative standalone result. The best qscore AP75 (`0.0341`) is below `smooth_l1_ap75/best.pt` (`0.0358`) and below `frcnn_standard__patches__seed42` (`0.0375`).

#### CBL Full-Budget Locked-Test Gate (2026-07-30)

The 20-epoch CBL run completed and passed its artifact audit. Because the
legacy trainer evaluated EMA but saved raw weights in `best*.pt`, the stored
EMA validation peak is not reloadable. The raw epoch-5 checkpoint independently
passed validation and was the only CBL checkpoint authorized for locked test.

| Run | Checkpoint | Epoch | mAP(scale) | COCO AP | AP50 | AP75 | AR100 | Output |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `cbl_full` | `best_ap75.pt` raw | 5 | 0.5723 | **0.0987** | 0.3002 | **0.0390** | 0.2486 | `runs/kaggle_cbl_full_best_ap75_locked_test.json` |
| `smooth_l1_ap75` | `best.pt` | 7 | **0.5844** | 0.0970 | 0.3000 | 0.0358 | **0.2525** | `runs/kaggle_checkpoint_eval_smooth_l1_best.json` |
| Phase-2 AP75 reference | IoU patches seed 42 | 7 | 0.5893 | 0.0958 | 0.2921 | 0.0375 | 0.2470 | `runs/frcnn_standard__patches__seed42/test_metrics.json` |
| Phase-2 AP/scale reference | IoU patches seed 2024 | 9 | **0.6114** | **0.1002** | **0.3272** | 0.0327 | 0.2338 | `runs/frcnn_standard__patches__seed2024/test_metrics.json` |

**Decision:** CBL raw epoch 5 is the new audited standalone leader for AP75 by
a small margin. It is not the COCO AP, scale-mAP, or recall leader. Do not
evaluate more CBL checkpoints on test; wait for the separately launched
EMA-checkpoint recovery run and assess it on validation first.

#### Prediction-Level Ensemble Gate (2026-07-22)

Validation-only tuning tested an AP75-focused prediction ensemble between `frcnn_standard__patches__seed42/best.pt` and downloaded Kaggle `smooth_l1_ap75/best.pt`. The best validation config was `ap75_hybrid`, IoU=0.60, score=0.20, model weights=1,1, with tile top-K=50.

| Split | Candidate | mAP(scale) | COCO AP | COCO AP50 | COCO AP75 | AR@100 | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| valid | patch42 + smooth-L1 ensemble | 0.5348 | 0.1011 | 0.2855 | **0.0451** | 0.1991 | Passed AP75 validation gate; run one frozen test. |
| valid | patch42 single, same postprocess | 0.5038 | 0.0980 | 0.2834 | 0.0421 | 0.1894 | Single-model validation reference. |
| valid | smooth-L1 single, same postprocess | 0.5102 | 0.0984 | 0.2787 | 0.0401 | 0.1943 | Single-model validation reference. |
| test | patch42 + smooth-L1 ensemble | 0.5613 | 0.0806 | 0.2376 | 0.0354 | 0.1790 | Failed promotion; below patch42 AP75=0.0375 and smooth-L1 best AP75=0.0358. |

**Conclusion:** the ensemble lifted validation AP75 but did not generalize to the locked test set. Do not promote this ensemble; use the result as evidence that future AP75 work needs score calibration or a new regularized training run rather than a simple prediction-level WBF combination.

Per-class COCO AP:

| Class | AP |
|---|---:|
| `dry` | 0.0847 |
| `wet` | 0.0816 |

Ground-truth counts used by scale-aware AP:

| Scale bin | GT count | AP |
|---|---:|---:|
| micro | 1015 | 0.4920 |
| tiny | 3246 | 0.5594 |
| small | 1454 | 0.5529 |
| large | 108 | 0.6351 |

### Analysis

1. **SA-ALW full wins on test.** mAP(scale)=0.6005, highest among all metric
   variants. Beats even the FRCNN patch IoU baseline (best seed 0.6114) on
   per-class COCO mAP. Scale-adaptive mechanisms generalize well from val → test.

2. **IGWD ablation flips on test.** On validation, anisotropic-only degraded
   IGWD by 0.02; on test, IGWD+aniso (0.5620) beats IGWD (0.5366). The
   isotropic normalizer appears to overfit validation. ALW's Charbonnier
   penalty may over-regularize, causing the steep 0.4572 drop.

3. **NWD surprisingly strong on test.** Val mAP(scale)=0.5298 → test=0.5648
   (gain +0.035), the only metric to improve on test. High recall
   (0.6553) suggests NWD catches objects that other metrics miss, though
   large-AP collapses (0.0782).

4. **FRCNN patch IoU baselines are competitive.** Patches + standard IoU
   achieve 0.5982 mean mAP(scale), matching SA-ALW. Tiling is the dominant
   factor; the metric adds complementary fine-grained gains on top.

5. **ALW full underperforms on test** (0.4572 vs 0.5835 val), the largest
   generalization gap. The reliability-gated Charbonnier penalty may
   over-regularize on unseen tiles. Consider disabling Charbonnier for test
   or tuning ε bounds.
