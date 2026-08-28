---
title: Tiny Object Architecture Improvement - 2026-05-31
type: analysis
created: 2026-05-31
updated: 2026-05-31
sources: [raw/sah_gd_hybrid_metrics_comparison.xlsx, eda/REPORT.md, wiki/analyses/tiny-object-metric-experiment-2026-05-31.md]
tags: [tiny-object-detection, architecture, improvement-plan]
---

## Tiny Object Architecture Improvement - 2026-05-31

## Question

After SAH-GD ablation completed with HARD_SWITCH winning (mAP=0.5770), what architectural changes should be prioritized to push beyond the current bottleneck?

## Current State

**Best metric baseline**: HARD_SWITCH_NWD_GCD with mAP(scale)=0.5770, AP_micro=0.2776, AP_tiny=0.5721, AP@75=0.0428

**Remaining bottlenecks**:
1. **Low AP@75** (0.0428) — strict localization still weak despite improvement from GCD (0.0415)
2. **Moderate AP_micro** (0.2776) — micro objects (<8px) lag behind tiny/small
3. **Dense scene handling** — 100 det/img is reasonable but could be tighter with better NMS calibration

## Dataset Context (from `eda/REPORT.md`)

- **92.1%** of objects have `sqrt(area) < 32px`
- **79.5%** have `sqrt(area) < 20px`
- **Bin (0,8)px** alone has **19,196 instances** (27% of dataset)
- **Average density**: 45 obj/img, max 730
- **Aspect ratio**: 81% are standing-pose (vertical)

## Proposed Improvements (Priority Order)

### 1. Add P2 / Stride-4 Features (HIGH PRIORITY)

**Why**: Current FPN uses P3-P6 (stride 8-64). The (0,8)px bin (19K instances) maps to **sub-pixel** on P3. P2 (stride 4) gives 2× spatial resolution for micro objects.

**How**:
- Enable P2 in FPN backbone (ResNet50 C2 → P2)
- Add P2 to RPN anchor generation
- Retune RPN/RoI head to handle P2 proposals
- **Risk**: +30-50% memory, slower training. May need gradient checkpointing or smaller batch size.

**Expected gain**: +5-10% AP_micro, +2-3% AP_tiny, better AP@75 for micro objects.

### 2. Retune Anchors for Micro/Tiny Bins (MEDIUM PRIORITY)

**Why**: Current anchors may not cover (0,8), [8,12), [12,20) bins well. RFLA uses Gaussian assignment but anchor coverage still matters for RPN recall.

**How**:
- Analyze current anchor sizes vs GT distribution (plot anchor-GT IoU heatmap by scale bin)
- Add smaller anchor sizes: e.g., `[4, 8, 12, 16, 24, 32, 48, 64]` instead of default `[32, 64, 128, 256, 512]`
- Increase anchor aspect ratios for standing pose: `[0.5, 0.75, 1.0, 1.5, 2.0]` (current: `[0.5, 1.0, 2.0]`)

**Expected gain**: +2-5% AP_micro, +1-2% AP_tiny, better proposal recall.

### 3. Calibrate NMS and Post-Processing (LOW PRIORITY, HIGH IMPACT)

**Why**: HARD_SWITCH achieves 100 det/img (vs GCD 119, NWD 214). This is good but can be tighter. AP@75 is low partly because duplicate boxes dilute precision.

**How**:
- **Tune NMS_SIM_THRESH** per scale bin: micro objects need looser threshold (0.25-0.30), large objects need tighter (0.10-0.15)
- **Add scale-aware NMS**: apply different thresholds based on box size
- **Tune SCORE_THRESH_TEST**: current 0.40 may be too high for micro objects (try 0.30-0.35)
- **Add duplicate suppression**: cluster boxes within 2-3px and keep highest score

**Expected gain**: +1-3% AP@75, -10-20 det/img, cleaner predictions.

### 4. Data Augmentation for Micro Objects (OPTIONAL)

**Why**: Micro objects are hardest to learn. Augmentation can help model generalize better.

**How**:
- **Mosaic augmentation**: paste 4 images into one (increases density, forces model to handle crowding)
- **Copy-paste micro objects**: duplicate small GT boxes within same image to balance scale distribution
- **Random crop with min-size constraint**: ensure crops contain at least N micro objects

**Expected gain**: +2-4% AP_micro, better generalization to dense scenes.

### 5. Two-Stage Refinement for Micro Objects (ADVANCED)

**Why**: Faster R-CNN RoI head uses 7×7 RoIAlign. For micro objects (<8px), this is too coarse. A second refinement stage can help.

**How**:
- **Cascade R-CNN**: add 2nd/3rd RoI head with stricter IoU thresholds (0.6, 0.7)
- **Scale-specific heads**: separate RoI head for micro objects with higher resolution (14×14 RoIAlign)
- **Iterative box refinement**: predict residual offsets in 2-3 iterations

**Expected gain**: +3-5% AP_micro, +2-3% AP@75, but +50-100% training time.

## Recommended Next Steps

**Immediate (1-2 weeks)**:
1. Enable P2 features + retune anchors → run HARD_SWITCH baseline
2. Calibrate NMS thresholds on validation set
3. Measure AP@75, AP_micro, det/img improvement

**If P2 + anchors succeed** (AP_micro > 0.30, AP@75 > 0.05):
4. Add mosaic augmentation
5. Experiment with scale-aware NMS

**If P2 + anchors plateau**:
6. Consider Cascade R-CNN or scale-specific heads (more engineering effort)

## Success Criteria

- **AP_micro > 0.30** (current: 0.2776)
- **AP@75 > 0.05** (current: 0.0428)
- **mAP(scale) > 0.60** (current: 0.5770)
- **Det/img < 90** (current: 100.08)

## Related Pages

- [[SAH-GD Hybrid Metrics Comparison]]
- [[Tiny Object Metric Experiment - 2026-05-31]]
- [[Tiny Object Detection Metrics]]
- [[RFLA]]
