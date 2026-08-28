---
title: RG-Robust ALW Implementation
type: source
created: 2026-06-02
updated: 2026-06-05
source_file: working/code/4_alw.ipynb, working/idea/6_rg_robust_alw.md, 4_alw_rg_robust.ipynb
sources: [working/code/4_alw.ipynb, working/idea/6_rg_robust_alw.md, 4_alw_rg_robust.ipynb]
tags: [alw, label-assignment, dynamic-topk, charbonnier, implementation]
---

## RG-Robust ALW Implementation

## Source

Combined implementation in `working/code/4_alw.ipynb` (rewritten 2026-06-02) merging the original ALW baseline with winning techniques discovered in `4_alw_rg_robust.ipynb`. Math described in `working/idea/6_rg_robust_alw.md`.

## Why the Reference Run Won

A variant run `4_alw_rg_robust.ipynb` scored `mAP(scale)=0.5549`, `AP_micro=0.3096` vs the original ALW baseline `mAP(scale)=0.3155`, `AP_micro=0.1410` — a **2.2× improvement on AP_micro**.

Root-cause analysis found the win came from **label assignment**, NOT the ALW formula:

1. **Anisotropic per-axis** (`Sx ≠ Sy`) — already present in the original ALW `_core_dist`, so NOT the source of the gap.
2. **Dynamic top-k by scale** — the dominant driver of AP_micro (+120%).
3. **Reliability-gated robust shape (Charbonnier)** — reduces gradient noise for micro boxes.

## Caveat: Different Datasets

⚠️ The two runs used **different datasets**:
- Original `4_alw`: dataset suggested as `hotrandinhnguyen/tod-dataset`
- Reference `4_alw_rg_robust`: `kurt54/sod-tinypeopleinsea`

So part of the 0.3155→0.5549 gap may be dataset difficulty. The merged version had to be re-run on the **same dataset** as other baselines for a fair comparison.

## Same-Dataset Rerun Result

The improved `working/code/4_alw.ipynb` rerun on the current TinyPerson-style dataset did **not** reproduce the reference gain:

| Metric | Value |
|--------|------:|
| Best mAP(scale) | 0.1822 |
| mAP@50 | 0.1256 |
| AP_micro | 0.1029 |
| AP_tiny | 0.2190 |
| AP_small | 0.1656 |
| AP_large | 0.1744 |
| COCO AP@50:75 | 0.0652 |
| COCO AP@75 | 0.0145 |
| AR@100 | 0.3589 |
| Avg det/img | 599.20 |

Interpretation: the assignment/shape changes increased duplicate/noisy predictions dramatically and did not recover AP. The earlier `0.5549` result should remain a non-comparable reference, not evidence that the merged ALW notebook is viable on this dataset.

## Three Algorithmic Improvements

### 1. Dynamic Top-k by Scale
Original: fixed `k=3` positives per GT.
New: micro=6, tiny=5, small=4, large=3, gated by `√(w·h)` thresholds (6/16/64px).

Micro GTs are "positive-starved" — giving them 2× positives (3→6) provides more learning signal. This directly explains the AP_micro jump.

### 2. Quality-Gated Assignment
Extra anchors (beyond base-k=3) are only accepted if `sim ≥ 0.60 × sim_best`. Prevents assigning distant anchors as positives, keeping label quality. Applied in both Pass-1 (original anchors) and Pass-2 (expanded anchors ×β=0.9).

### 3. Reliability-Gated Robust Shape (Charbonnier)
Original shape term `[ln(wa/wb)]² + [ln(ha/hb)]²` amplifies measurement noise for micro boxes (±1px ≈ ±15-50% size error).

New shape term gates by GT size:
- `g(b) = clamp(√(wb·hb)/16, 0, 1)` — micro GT → g≈0
- `λ(b) = 0.35 + 0.65·g(b)^1.5` — micro GT trusts shape less
- `ε(b) = 1e-3 + 0.35·(1-g(b))` — micro GT uses smoother Charbonnier
- `shp = λ·[(√(rw²+ε²)-ε) + (√(rh²+ε²)-ε)]`

Charbonnier behaves like L2 near zero (smooth gradient) and L1 far from zero (sharp), with the transition point controlled by `ε`.

## What Was Kept From Original (Believed Still Correct)

| Component | Value | Rationale |
|-----------|-------|-----------|
| MIN/MAX_SIZE | 512/896 | High resolution good for tiny (reference used lower 416/640 but still won → keep high res) |
| GRAD_ACCUM | 4 (eff batch 16) | Larger batch = stable gradients |
| **MetricRoIHeads + metric-NMS** | NMS_SIM=0.15 | Reference used loose IoU-NMS → **296 det/img** over-detection. Sharp metric-NMS suppresses duplicates better |
| SCORE_THRESH_TEST | 0.40 | Reduce over-detection (reference 0.30 → more junk boxes) |
| metric box-loss (1-sim) + warmup | kept | Direct metric optimization; warmup stabilizes start |
| Anchors | EDA-tuned | No giant anchors, vertical aspect ratios |

This is the key design decision: the merged version takes the reference's **assignment** improvements but rejects its **post-processing** (loose NMS, low score thresh) which caused 296 det/img.

## Original Expected Results, Now Falsified

- AP_micro: expected `0.14 → ~0.30`; observed `0.1029`.
- Over-detection: target `<150 det/img`; observed `599.20 det/img`.
- Stability: expected lower val_loss; observed `val_loss=0.6340`.
- AP@75: expected still bottleneck; observed `0.0145`, worse than the original ALW run.

## Validation

All 9 cells compile. Runtime test confirmed:
- `metric_sim_matrix` returns [N,M] in range (0,1]
- `get_dynamic_k` returns correct k per scale (micro=5-6, large=4)
- Assignment produces sensible per-GT positive counts
- Kaggle-compliant: no local imports, `torch.amp.autocast`, self-contained

## Related Pages

- [[Anisotropic Log-Wasserstein Distance (ALW)]]
- [[Tiny Object Detection Metrics]]
- [[Tiny Object Architecture Improvement - 2026-05-31]]
- [[RFLA]]
