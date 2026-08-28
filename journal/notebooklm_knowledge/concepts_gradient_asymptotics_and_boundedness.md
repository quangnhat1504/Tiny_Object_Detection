---
title: "Gradient Asymptotics and Non-Vanishing Bound"
type: "concept"
created: "2026-08-23"
updated: "2026-08-23"
sources:
  - "journal/manuscript/main.tex"
tags:
  - "theory"
  - "gradients"
  - "proof"
---

# Gradient Asymptotics and Non-Vanishing Bound

## 1. Theorem Formulation
**Theorem 1 (Non-Vanishing Gradient Bound under Disjoint Misalignment)**:
Let target box $B$ have scale $s_B \to 0$. For any predicted box $A$ with disjoint intersection $\text{IoU}(A, B) = 0$, the gradient of the Homotopy loss $\mathcal{L}_{\text{H-WIoU}} = 1 - \mathcal{S}_{\text{H-WIoU}}$ with respect to box parameter $\theta = [x_a, y_a, w_a, h_a]$ satisfies:

$$\|\nabla_\theta \mathcal{L}_{\text{H-WIoU}}\| = \mathcal{O}(1) > 0$$

whereas the standard IoU gradient collapses completely:
$$\|\nabla_\theta \mathcal{L}_{\text{IoU}}\| \equiv 0$$

## 2. Mathematical Proof
For disjoint boxes $A$ and $B$, $\text{IoU}(A, B) = 0$. Under the continuous Homotopy regularization:
$$\mathcal{S}_{\text{H-WIoU}}(A, B) = \exp\left(-(1 - \gamma(s_B))\mathcal{D}_{\mathcal{W}}^2(A, B)\right)$$

Taking the partial derivative with respect to spatial coordinate $x_a$:
$$\frac{\partial \mathcal{L}_{\text{H-WIoU}}}{\partial x_a} = -\frac{\partial \mathcal{S}_{\text{H-WIoU}}}{\partial x_a} = (1 - \gamma(s_B))\mathcal{S}_{\text{H-WIoU}}(A, B) \cdot \frac{2(x_a - x_b)}{\bar{w}_{ab}^2}$$

Under microscopic scale $s_B < \sigma_0$, we have $\gamma(s_B) < 0.5$ and $(1 - \gamma(s_B)) > 0.5$.
For any finite spatial offset $\Delta x = x_a - x_b \neq 0$:
$$\left| \frac{\partial \mathcal{L}_{\text{H-WIoU}}}{\partial x_a} \right| = \frac{2 |\Delta x|}{\bar{w}_{ab}^2}(1 - \gamma(s_B))\exp\left(-(1 - \gamma(s_B))\mathcal{D}_{\mathcal{W}}^2\right) > 0$$

As $\Delta x \to 0$, the gradient exhibits smooth linear restoration $\propto \Delta x / \bar{w}_{ab}^2$, maintaining stable $C^\infty$ convergence and avoiding numerical oscillations.
