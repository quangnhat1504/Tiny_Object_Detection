---
title: "Stage 2: Bounded Homotopy Box Loss in RoI Head"
type: "concept"
created: "2026-08-23"
updated: "2026-08-23"
sources:
  - "journal/manuscript/main.tex"
  - "common/metrics/iou.py"
tags:
  - "roi-head"
  - "loss"
  - "regression"
---

# Stage 2: Bounded Homotopy Box Loss in RoI Head

## 1. Loss Formulation
In Stage 2 (RoI Head), candidate Region-of-Interest features pooled via $7\times 7$ RoIAlign are supervised using the bounded Homotopy regression objective:

$$\mathcal{L}_{\text{H-WIoU}}(P_i, G_i) = 1 - \mathcal{S}_{\text{H-WIoU}}(P_i, G_i)$$

where $P_i$ is the predicted box and $G_i$ is the assigned ground-truth box.

## 2. Boundedness & Stability Properties
1. **Strict Bounded Range**: Because $\text{IoU} \in [0, 1]$ and $\mathcal{D}_{\mathcal{W}}^2 \ge 0$, we have:
   $$\mathcal{S}_{\text{H-WIoU}} \in [0, 1] \implies \mathcal{L}_{\text{H-WIoU}} \in [0, 1]$$
   This eliminates loss explosion without needing heuristic gradient clipping ($L_1 / \text{Smooth-}L_1$ thresholds).
2. **Scale-Aware Boundary Tightening**:
   * For microscopic instances, the loss penalizes centroid and variance divergence with smooth Gaussian gradients.
   * For larger instances ($s > 20\text{px}$), the $(1 - \mathcal{S}_{\text{H-WIoU}})$ loss converges to $(1 - \text{IoU})$, enforcing tight boundary overlap and maximizing strict $\text{AP}_{75}$ localization metrics.
