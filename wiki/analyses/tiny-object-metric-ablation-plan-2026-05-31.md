---
title: Tiny Object Metric Ablation Plan - 2026-05-31
type: analysis
created: 2026-05-31
updated: 2026-05-31
sources: [working/idea/5_sah_gd.md, working/code/README_sah_ablation.md, working/code/5_adaptive_nwd.ipynb, working/code/6_hard_switch_nwd_gcd.ipynb, working/code/7_sah_gd_soft_blend.ipynb, working/code/8_sah_gd_scale_topk.ipynb, raw/sah_gd_hybrid_metrics_comparison.xlsx]
tags: [tiny-object-detection, ablation, kaggle, sah-gd, completed]
---

## Tiny Object Metric Ablation Plan - 2026-05-31

## Question

How should the proposed [[Scale-Adaptive Hybrid Gaussian Distance (SAH-GD)]] be tested on Kaggle before deeper architectural changes?

## Status: COMPLETED

All four ablation runs completed successfully. Results ingested from [[SAH-GD Hybrid Metrics Comparison]].

## Ablation Results

| Notebook | Purpose | mAP(scale) | AP_micro | AP_tiny | AP@75 | Det/img |
|----------|---------|------------|----------|---------|-------|---------|
| `5_adaptive_nwd.ipynb` | Adaptive NWD only | 0.5671 | 0.2395 | 0.5617 | 0.0345 | 94.63 |
| `6_hard_switch_nwd_gcd.ipynb` | Hard switch NWD/GCD | **0.5770** | 0.2776 | 0.5721 | 0.0428 | 100.08 |
| `7_sah_gd_soft_blend.ipynb` | SAH-GD soft blend | 0.5752 | 0.2758 | 0.5686 | 0.0447 | 100.68 |
| `8_sah_gd_scale_topk.ipynb` | SAH-GD + scale top-k | 0.5768 | **0.2947** | **0.5810** | **0.0453** | 101.46 |

## Decision Outcome

**HARD_SWITCH_NWD_GCD is the recommended baseline** per the decision rule:
- Highest overall mAP(scale): 0.5770
- Beats GCD baseline (0.5522) by +4.5%
- Simpler than soft blend, no sigmoid tuning needed
- Reasonable detection count (100 det/img vs GCD's 119)

**Soft blend did NOT beat hard switch**, so the novelty claim should be reframed around:
1. Adaptive `C(s)` normalization (proven useful by all variants beating GCD)
2. Scale-aware top-k assignment (SCALE_TOPK wins AP_micro/AP_tiny/AP@75)

## Next Architecture Improvements

SAH-GD ablation is complete. The bottleneck is now **feature resolution and proposal coverage**, not metric formula. See [[Tiny Object Architecture Improvement - 2026-05-31]] for next steps.

## Related Pages

- [[Scale-Adaptive Hybrid Gaussian Distance (SAH-GD)]]
- [[SAH-GD Hybrid Metrics Comparison]]
- [[Tiny Object Metric Experiment - 2026-05-31]]
- [[Tiny Object Detection Metrics]]
