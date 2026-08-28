---
title: SAH-GD Hybrid Metrics Comparison
type: source
created: 2026-05-31
updated: 2026-06-05
source_file: raw/sah_gd_hybrid_metrics_comparison.xlsx
sources: [raw/sah_gd_hybrid_metrics_comparison.xlsx]
tags: [experiment-results, sah-gd, hybrid-metrics, ablation]
---

## SAH-GD Hybrid Metrics Comparison

## Source

Excel file containing ablation results for four SAH-GD variants tested on the TinyPerson dataset with Faster R-CNN + RFLA architecture. All core SAH-GD runs completed 12 epochs with identical base configuration except for the metric blending strategy. Notebooks 10-12 (`HARD_SWITCH_P2_DUAL`, `SCALE_TOPK_P2`, `HARD_SWITCH_P2_TOPK_DUAL`) were added later as P2 follow-ups and are not like-for-like metric-only ablations.

## Four Variants Tested

### 1. ADAPTIVE_NWD (Baseline)
- **Formula**: Pure adaptive NWD with scale-dependent normalization constant
- **Config**: `C(s) = sqrt((λ·s̄)² + C_min²)` where `λ=1.0`, `C_min=4.0`, `s̄ = 0.5·(√(wa·ha) + √(wb·hb))`
- **Distance**: `d = W2 / C(s̄)` where `W2 = sqrt(dx² + dy² + ((wa-wb)/2)² + ((ha-hb)/2)²)`
- **Purpose**: Test whether adaptive normalization alone is sufficient

### 2. HARD_SWITCH_NWD_GCD (Recommended)
- **Formula**: Hard threshold switch between adaptive NWD and GCD
- **Config**: `τ=8.0px`, `λ=1.0`, `C_min=4.0`, `β=4.0`
- **Distance**: `d = (s̄ < τ) ? d_NWD : d_GCD` where `s̄ = 0.5·(√(wa·ha) + √(wb·hb))`
- **Purpose**: Test whether a simple non-smooth rule beats soft blending

### 3. SAH_GD_SOFT_BLEND
- **Formula**: Smooth sigmoid blend between adaptive NWD and GCD
- **Config**: `τ=8.0px`, `K=2.0`, `λ=1.0`, `C_min=4.0`, `β=4.0`
- **Distance**: `d = (1-α)·d_NWD + α·d_GCD` where `α = sigmoid((s̄-τ)/K)`
- **Purpose**: Test the main proposed differentiable transition

### 4. SAH_GD_SCALE_TOPK
- **Formula**: Same as SOFT_BLEND + scale-aware top-k positive selection
- **Config**: Same as SOFT_BLEND, plus micro GT gets more positive anchors
- **Distance**: Same as SOFT_BLEND
- **Purpose**: Test whether micro GT needs more positives to improve AP_micro

## Results Summary

| Method | mAP(scale) | mAP@50 | AP_micro | AP_tiny | AP_small | AP_large | COCO AP@75 | AR@100 | Det/img | Inference (s/img) |
|--------|------------|--------|----------|---------|----------|----------|------------|--------|---------|-------------------|
| **HARD_SWITCH_NWD_GCD** | **0.5770** | **0.3517** | 0.2776 | 0.5721 | 0.6600 | **0.7620** | 0.0428 | 0.3818 | 100.08 | 4.86 |
| SAH_GD_SCALE_TOPK | 0.5768 | 0.3592 | **0.2947** | **0.5810** | 0.6487 | 0.7059 | **0.0453** | **0.3869** | 101.46 | 4.66 |
| SAH_GD_SOFT_BLEND | 0.5752 | 0.3537 | 0.2758 | 0.5686 | **0.6604** | 0.7559 | 0.0447 | 0.3827 | 100.68 | 5.08 |
| ADAPTIVE_NWD | 0.5671 | 0.3354 | 0.2395 | 0.5617 | 0.6610 | 0.7296 | 0.0345 | 0.3551 | **94.63** | **3.25** |
| HARD_SWITCH_P2_DUAL | 0.4516 | 0.2452 | 0.2806 | 0.4901 | 0.4734 | 0.2852 | 0.0129 | 0.2886 | **72.54** | 5.04 |
| SCALE_TOPK_P2 | 0.4522 | 0.2493 | 0.2821 | 0.4971 | 0.4666 | 0.2823 | 0.0121 | 0.2956 | 75.02 | 4.94 |
| HARD_SWITCH_P2_TOPK_DUAL | 0.4724 | 0.2597 | 0.3151 | 0.5151 | 0.4863 | 0.2903 | 0.0145 | 0.2959 | **70.45** | 5.07 |

## Key Findings

1. **HARD_SWITCH wins overall**: Highest mAP(scale), best AP_large, balanced performance, reasonable detection count
2. **SCALE_TOPK wins micro/tiny**: Best AP_micro (+6.2% vs HARD_SWITCH), best AP_tiny (+1.6%), best AP@75 (localization), best AR@100 (recall)
3. **SOFT_BLEND has no advantage**: No metric leads, slowest inference (5.08s/img), no clear benefit over hard switch
4. **ADAPTIVE_NWD is fastest but weakest**: Lowest AP_micro (0.2395), lowest AP@75 (0.0345), fewest detections (94.63/img)
5. **P2 follow-ups reduced detections but underperformed the earlier P2 baseline**: `HARD_SWITCH_P2_TOPK_DUAL` is the best of notebooks 10-12 (`mAP(scale)=0.4724`, `AP_micro=0.3151`, `70.45 det/img`), but it is still below the earlier P2 result and leaves strict localization weak (`COCO AP@75=0.0145`).

## Comparison to Previous Baselines

From [[Tiny Object Metric Experiment - 2026-05-31]], the previous best was GCD with `mAP(scale)=0.5522`, `AP_micro=0.2582`, `AP_tiny=0.5437`, `det/img=119.38`.

**All four SAH-GD variants beat the GCD baseline**:
- HARD_SWITCH: +4.5% mAP (0.5770 vs 0.5522)
- SCALE_TOPK: +14.1% AP_micro (0.2947 vs 0.2582), +6.9% AP_tiny (0.5810 vs 0.5437)
- Detection counts improved: 94-101 det/img vs 119 (less duplicate noise)

## Bottleneck Analysis

**AP@75 remains low** (0.0345-0.0453) but **SCALE_TOPK achieved 0.0453** — the highest strict localization score across all experiments (previous best: GCD=0.0415, NWD=0.0376). This suggests:
- Localization is improving but still a bottleneck
- Scale-aware top-k helps micro-object localization more than the metric formula itself
- Further gains require architectural changes (feature resolution, anchor coverage)

## Recommendation

**Use HARD_SWITCH_NWD_GCD as the main baseline** for balanced performance and simplicity. The hard threshold is easier to interpret, faster than soft blend, and achieves the best overall mAP with strong large-object performance.

**Consider SCALE_TOPK for micro-object-focused applications** where AP_micro and AP_tiny are critical metrics.

## Related Pages

- [[Scale-Adaptive Hybrid Gaussian Distance (SAH-GD)]]
- [[Tiny Object Metric Experiment - 2026-05-31]]
- [[Tiny Object Detection Metrics]]
- [[NWD]]
- [[GCD]]
