---
title: CBL SNIP-Like Scale-Normalized Training Local Gate - 2026-07-31
type: analysis
created: 2026-07-31
updated: 2026-07-31
sources: [cvpr:2018-snip, neurips:2018-sniper, common/model.py, scripts/train_frcnn_metric.py, scripts/test_snip_scale_supervision.py, runs/cbl_iterative_train_snip_multiscale_best_ap75_valid_reload.json]
tags: [cbl, snip, multi-scale-training, selective-supervision, ap75, negative-result]
---

# CBL SNIP-Like Scale-Normalized Training Local Gate - 2026-07-31

## Question

Can correct ignored-object supervision recover the performance lost by naive
stochastic multi-scale training while preserving the scale-960 benefit seen at
inference?

## Method Contract

SNIP backpropagates an object only at image resolutions where its transformed
scale is valid. SNIPER preserves the same scale-normalization principle using
scale-specific context chips. Both papers require out-of-range objects to be
ignored, not relabeled as background.

Primary references:

- [SNIP, CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Singh_An_Analysis_of_CVPR_2018_paper.html)
- [SNIP, arXiv](https://arxiv.org/abs/1711.08189)
- [SNIPER, NeurIPS 2018](https://proceedings.neurips.cc/paper_files/paper/2018/hash/166cee72e93a992007a89b39eb29628b-Abstract.html)
- [SNIPER paper](https://proceedings.neurips.cc/paper_files/paper/2018/file/166cee72e93a992007a89b39eb29628b-Paper.pdf)

The bounded local implementation keeps the failed multi-scale run's seed,
optimizer-step count, and transform choices `[640,800,960]/1200`. It changes
only scale-valid supervision:

| Training short side | Valid transformed sqrt-area |
|---:|---:|
| 640 | `[20, infinity)` px |
| 800 | `[12.5, 50]` px |
| 960 | `[0, 30]` px |

The ranges are the project's micro/tiny/small boundaries translated through
the selected transform. They overlap, so every training GT is valid at one or
two scales.

## Correct Ignore Semantics

- The transform records the selected scale and a per-GT validity mask after
  resizing.
- RPN positives are assigned only from valid GT. Anchors with invalid-GT IoU
  at least `0.40` are ignored when invalid overlap exceeds valid overlap.
- RoI assignment retains all GT, then labels proposals outside the active
  transformed size range as `-1` so the sampler ignores them.
- Evaluation and inference remain fixed at `640/800`; SNIP state changes only
  training supervision.
- The wrappers call unbound class methods so deep-copied EMA models do not
  retain references to the raw training model.

Synthetic CUDA verification sampled all three scales, reproduced the expected
RPN and RoI ignore labels, completed a finite backward pass, and confirmed
that an EMA copy evaluates independently. The existing iterative-CBL
train/inference/reload regression test also passed.

## Supervision Audit

The training tiles contain `131,836` clipped GT boxes:

| Band | Count |
|---|---:|
| Micro `<8` px | 37,356 |
| Tiny `8-16` px | 56,415 |
| Small `16-32` px | 27,886 |
| Large `>=32` px | 10,179 |

Per-scale valid fractions are `28.87%` at 640, `63.94%` at 800, and `71.13%`
at 960. No GT is uncovered. A six-step real-data optimizer audit observed all
three scales, non-zero RPN/RoI ignore counts, finite losses, and valid
background negatives even when a sampled image had zero valid GT.

## Two-Epoch Result

| Setting | Epoch | AP | AP50 | AP75 | AR100 | Seconds |
|---|---:|---:|---:|---:|---:|---:|
| Fixed-scale baseline | 1 | 0.1161 | 0.3328 | 0.0498 | 0.2613 | 639.98 |
| Naive multi-scale | 1 | 0.1141 | 0.3295 | 0.0436 | 0.2590 | 2211.56 |
| SNIP-like stochastic | 1 | 0.1041 | 0.3014 | 0.0384 | 0.2358 | 2230.27 |
| Fixed-scale baseline | 2 | **0.1270** | **0.3614** | **0.0573** | **0.2761** | 634.06 |
| Naive multi-scale | 2 | 0.1048 | 0.3197 | 0.0401 | 0.2636 | 1897.37 |
| SNIP-like stochastic | 2 | 0.1051 | 0.3060 | 0.0441 | 0.2490 | 2124.42 |

Independent reload of SNIP-like epoch 2 reproduces
AP/AP50/AP75/AR100=`0.1052/0.3060/0.0440/0.2490`. The checkpoint is raw epoch
2, declares matching raw metric provenance, and records the transform/range
contract.

Against independently reloaded fixed-scale epoch 2:

| Metric | Fixed scale | SNIP-like | Delta |
|---|---:|---:|---:|
| AP | 0.1269 | 0.1052 | -0.0217 |
| AP50 | 0.3612 | 0.3060 | -0.0552 |
| AP75 | 0.0572 | 0.0440 | -0.0132 |
| AR100 | 0.2758 | 0.2490 | -0.0268 |
| Legacy micro AP | 0.3400 | 0.1075 | -0.2325 |
| Legacy tiny AP | 0.6197 | 0.5427 | -0.0770 |
| Legacy small AP | 0.6267 | 0.6555 | +0.0288 |
| Legacy large AP | 0.6974 | 0.5083 | -0.1891 |

The isolated small-band gain does not compensate for the large losses in
micro supervision, total AP, strict localization, and recall.

## Interpretation and Decision

Negative performance gate. Do not launch this configuration on Kaggle, inspect
the locked test set, or sweep validity thresholds.

This experiment is a controlled **stochastic SNIP-like approximation**, not
full SNIP or SNIPER. Full SNIP presents every image at multiple pyramid scales;
this equal-step local gate selects only one scale per image. An object valid at
one of three scales therefore receives fewer expected positive updates. The
result rejects this compute-matched stochastic formulation but does not claim
that full image-pyramid SNIP fails.

The implementation remains an opt-in reproducible ablation surface at commit
`e54c9ec` on `cbl-iterative-depth-20260731`.

## Artifacts

- `runs/sa_alw_full__cbl__irtw0.5ir1s0.3__la_loss__seed42__cbl_iterative_train_snip_ms640_800_960_local_gate/metrics.csv`
- `runs/sa_alw_full__cbl__irtw0.5ir1s0.3__la_loss__seed42__cbl_iterative_train_snip_ms640_800_960_local_gate/best_ap75.pt`
- `runs/cbl_iterative_train_snip_multiscale_best_ap75_valid_reload.json`

## Related Pages

- [[CBL Stochastic Multi-Scale Training Local Gate - 2026-07-31]]
- [[CBL Transform-Scale TTA Local Gate - 2026-07-31]]
- [[Trainable Iterative CBL Local Gate - 2026-07-31]]
