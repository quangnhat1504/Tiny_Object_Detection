---
title: "Stage 1: Homotopy Label Assignment (HLA) in RPN"
type: "concept"
created: "2026-08-23"
updated: "2026-08-23"
sources:
  - "journal/manuscript/main.tex"
  - "common/model.py"
tags:
  - "rpn"
  - "label-assignment"
  - "architecture"
---

# Stage 1: Homotopy Label Assignment (HLA) in RPN

## 1. Problem with Standard IoU Assignment
In standard Faster R-CNN RPN:
* Anchors with $\text{IoU} \ge 0.7$ are designated as Positive.
* Anchors with $\text{IoU} < 0.3$ are designated as Negative.
* For microscopic targets ($s < 8\text{px}$), the discrete overlap with fixed anchor grids rarely exceeds $0.2$, meaning almost all microscopic ground truths get zero positive anchors ($> 70\%$ starvation rate).

## 2. Homotopy Label Assignment Mechanism
In H-WIoU Stage 1:
1. Candidate anchor proposals $\mathcal{A} = \{A_i\}_{i=1}^M$ are evaluated against ground truths $\mathcal{G} = \{G_j\}_{j=1}^K$ using the continuous Homotopy similarity matrix:
   $$\mathbf{S}_{i,j} = \mathcal{S}_{\text{H-WIoU}}(A_i, G_j)$$
2. For each ground truth $G_j$, the effective receptive field dynamically expands as a function of target scale $s_j$:
   * For $s_j < \sigma_0$, the Gaussian Wasserstein envelope spans multiple anchor strides, assigning positive labels to anchors possessing strong optimal transport potential.
   * For $s_j \ge \sigma_0$, the assignment criterion smoothly contracts to strict IoU boundaries.
3. Top-$k$ dynamic assignment assigns the highest-scoring candidate anchors to each tiny instance, lifting positive candidate survival rate from **$0.18 \to 0.94$** ($5.2\times$ gain).
