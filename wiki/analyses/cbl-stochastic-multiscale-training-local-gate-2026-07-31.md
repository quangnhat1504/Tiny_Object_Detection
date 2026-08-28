---
title: CBL Stochastic Multi-Scale Training Local Gate - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources: [cvpr:2018-snip, neurips:2018-sniper, common/model.py, scripts/train_frcnn_metric.py, runs/cbl_iterative_train_multiscale_best_ap75_valid_reload.json]
tags: [cbl, multi-scale-training, snip, localization, ap75, negative-result]
---

# CBL Stochastic Multi-Scale Training Local Gate - 2026-07-31

## Question

Can stochastic transform scales absorb part of the positive scale-960 TTA gain
into the trainable iterative CBL weights while preserving fixed-scale
validation?

## Evidence and Design

SNIP reports that naive multi-scale training is not sufficient by itself and
instead backpropagates objects only when their scale is valid for the current
image resolution. SNIPER applies the same scale-normalization idea through
appropriate-scale training chips. This first bounded gate intentionally tests
the cheaper naive prerequisite before implementing ignored-object supervision:

- baseline: trainable iterative CBL, refinement loss weight `0.50`, one
  inference refinement pass at score `>=0.30`;
- seed `42`, raw weights, batch size `4`, two epochs;
- the existing copy-paste and tiny-tile oversampling settings are unchanged;
- training shorter-side choices are `[640, 800, 960]`, with maximum side
  `1200`;
- validation is explicitly reset to the project default `640/800`.

Primary references:

- [SNIP, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Singh_An_Analysis_of_CVPR_2018_paper.html)
- [SNIPER, NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/166cee72e93a992007a89b39eb29628b-Abstract.html)

## Technical Gate

- Torchvision `0.26.0+cu128` chooses from `transform.min_size` during training.
- The training script now records train and evaluation transform sizes in each
  checkpoint and resets evaluation to `(640,)/800` before validation.
- A four-step CUDA optimizer smoke sampled all three scales and produced finite
  losses `6.2776, 6.2609, 6.2983, 6.2914`.
- A fixed `960/1200` batch-size-4 backward pass peaked at `7.921 GiB`
  allocated and `8.154 GiB` reserved on the RTX 5070 Ti.
- The complete two-epoch run had no OOM, NaN, skipped batch, or native crash.

Windows WDDM still paged on some large-shape batches. The existing
`TOD_EMPTY_CACHE_EVERY=1` workaround was retained for comparability with the
local baseline.

## Two-Epoch Result

| Setting | Epoch | AP | AP50 | AP75 | AR100 | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-scale baseline | 1 | 0.1161 | 0.3328 | 0.0498 | 0.2613 | 639.98 |
| Multi-scale `[640,800,960]` | 1 | 0.1141 | 0.3295 | 0.0436 | 0.2590 | 2211.56 |
| Fixed-scale baseline | 2 | **0.1270** | **0.3614** | **0.0573** | **0.2761** | 634.06 |
| Multi-scale `[640,800,960]` | 2 | 0.1048 | 0.3197 | 0.0401 | 0.2636 | 1897.37 |

The multi-scale model peaks at epoch 1 and then declines. Its independently
reloaded best-AP75 checkpoint reproduces AP/AP50/AP75/AR100 =
`0.1141/0.3294/0.0436/0.2590`. The checkpoint is raw epoch 1, its model source
matches the stored metric source, and its config records train
`[640,800,960]/1200` plus evaluation `640/800`.

Against the independently reloaded fixed-scale epoch-2 baseline, the best
multi-scale checkpoint changes:

| Metric | Fixed scale | Multi-scale | Delta |
|---|---:|---:|---:|
| AP | 0.1269 | 0.1141 | -0.0128 |
| AP50 | 0.3612 | 0.3294 | -0.0318 |
| AP75 | 0.0572 | 0.0436 | -0.0136 |
| AR100 | 0.2758 | 0.2590 | -0.0168 |
| Legacy micro AP | 0.3400 | 0.2592 | -0.0808 |
| Legacy tiny AP | 0.6197 | 0.5720 | -0.0477 |
| Legacy small AP | 0.6267 | 0.6596 | +0.0329 |
| Legacy large AP | 0.6974 | 0.6136 | -0.0838 |

The isolated small-band gain does not compensate for the losses in strict
localization, recall, micro/tiny objects, and overall AP. The result also costs
about three times the local epoch runtime.

## Decision

Negative performance gate. Do not launch this configuration on Kaggle and do
not inspect the locked test set. Keep the transform controls as a reproducible
ablation surface, but do not sweep more naive scale tuples.

The reusable transform controls are pushed on branch
`cbl-iterative-depth-20260731` at commit `7e01340`.

The result is consistent with the SNIP motivation that indiscriminate
multi-scale training can expose objects at unsuitable resolutions. It does not
prove that selective scale-normalized training will work here: this is one
seed, two epochs, and a fixed local budget.

The next bounded train-time route is SNIP-like selective supervision. It must
mark anchors and proposals overlapping out-of-range objects as ignored; simply
dropping those ground-truth boxes would turn them into false background and is
not a valid implementation.

## Artifacts

- `runs/sa_alw_full__cbl__irtw0.5ir1s0.3__la_loss__seed42__cbl_iterative_train_ms640_800_960_local_gate/metrics.csv`
- `runs/sa_alw_full__cbl__irtw0.5ir1s0.3__la_loss__seed42__cbl_iterative_train_ms640_800_960_local_gate/best_ap75.pt`
- `runs/cbl_iterative_train_multiscale_best_ap75_valid_reload.json`
