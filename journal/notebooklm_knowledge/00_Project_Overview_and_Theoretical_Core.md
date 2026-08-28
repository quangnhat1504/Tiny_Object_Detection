---
title: "Journal Overview: Homotopy Wasserstein-IoU Framework"
type: "overview"
created: "2026-08-23"
updated: "2026-08-23"
sources:
  - "journal/manuscript/main.tex"
  - "common/metrics/iou.py"
tags:
  - "journal"
  - "overview"
  - "h-wiou"
---

# Journal Overview: Homotopy Wasserstein-IoU Framework

## 1. Problem Statement: The Microscopic Dilemma
Detecting tiny and microscopic objects ($s < 8\text{px}$) in high-resolution aerial and maritime imagery is plagued by a fundamental mathematical breakdown:
1. **Intersection-over-Union (IoU) Discontinuity**: A positional perturbation of merely $1\text{--}2\text{px}$ between bounding boxes drops the discrete intersection area to zero ($\text{IoU} = 0$). Consequently, the regression gradient vanishes ($\nabla_\theta \mathcal{L}_{\text{IoU}} \equiv 0$), leading to severe anchor starvation during Region Proposal Network (RPN) training ($> 70\%$ miss rate).
2. **Optimal Transport Boundary Blurring**: While distance-based metrics such as Normalized Wasserstein Distance (NWD) model bounding boxes as 2D Gaussians and provide continuous non-zero gradients, their isotropic nature acts as a low-pass spatial filter. This induces significant localization drift on medium and normal scale objects, degrading $\text{AP}_{75}$ and overall boundary precision.

## 2. Core Solution: Homotopy Wasserstein-IoU (H-WIoU)
To resolve this trade-off without introducing auxiliary networks or inference latency, we introduce **Homotopy Wasserstein-IoU (H-WIoU)**. Drawing upon topological homotopy theory, H-WIoU constructs a continuous, scale-adaptive map:

$$\mathcal{S}_{\text{H-WIoU}}(A, B) = [\text{IoU}(A, B)]^{\gamma(s_B)} \cdot \exp\left(-(1 - \gamma(s_B))\mathcal{D}_{\mathcal{W}}^2(A, B)\right)$$

where $\gamma(s) = \frac{s^2}{s^2 + \sigma_0^2} \in [0, 1)$ smoothly transitions between the two regimes:
* **Micro Scale Regime ($s < 8\text{px}$)**: $\gamma(s) \to 0$, activating pure 2-Wasserstein optimal transport $\mathcal{W}_2$ to pull disjoint candidate anchors with non-vanishing force $\|\nabla \mathcal{L}\| = \mathcal{O}(1)$.
* **Normal Scale Regime ($s > 20\text{px}$)**: $\gamma(s) \to 1$, smoothly recovering sharp discrete IoU boundaries to ensure strict $\text{AP}_{75}$ localization fidelity.

## 3. Two-Stage Integration Pipeline
* **Stage 1 (RPN Homotopy Label Assignment)**: Dynamically expands effective receptive fields for microscopic ground truths, increasing average positive anchor survival rate from $0.18 \to 0.94$ per target ($5.2\times$ gain).
* **Stage 2 (RoI Head Bounded Loss)**: Supervises candidate boxes with the bounded objective $\mathcal{L}_{\text{H-WIoU}} = 1 - \mathcal{S}_{\text{H-WIoU}} \in [0, 1]$, preventing gradient explosion while enforcing sub-pixel boundary fit.

## 4. Key Empirical Highlights
* **TinyPerson (Fair-20 Protocol)**:
  * $\text{mAP}_{50}: 0.4027 \to \mathbf{0.4618}$ ($+5.91\%$ absolute gain over Faster R-CNN baseline).
  * $\text{AP}_{\text{micro}}: 0.3307 \to \mathbf{0.3616}$ ($+3.09\%$ gain).
  * $\text{AP}_{\text{tiny}}: 0.6124 \to \mathbf{0.7144}$ ($+10.20\%$ gain).
  * $\text{AP}_{75}: 0.0719 \to \mathbf{0.0729}$ (outperforming NWD $0.0669$ by $+9.0\%$ relative).
* **AI-TOD-v2 (8-Class Aerial Matrix)**:
  * $\text{AP}_{vt}: 1.9\% \to \mathbf{12.3\%}$ ($+6.4\times$ improvement).
  * $\text{AP}_{50}: 26.3\% \to \mathbf{46.2\%}$ ($+19.9\%$ gain).
* **Inference Overhead**: $+0\text{MB}$ parameter bloat, $100\%$ standard Faster R-CNN inference speed ($> 54\text{ FPS}$).
