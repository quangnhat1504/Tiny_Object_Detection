---
title: "Homotopy Wasserstein-IoU (H-WIoU) Theory"
type: "concept"
created: "2026-08-23"
updated: "2026-08-23"
sources:
  - "journal/manuscript/main.tex"
  - "common/metrics/iou.py"
tags:
  - "theory"
  - "homotopy"
  - "wasserstein"
  - "iou"
---

# Homotopy Wasserstein-IoU (H-WIoU) Theory

## 1. Mathematical Definition
Let bounding boxes $A = (x_a, y_a, w_a, h_a)$ and $B = (x_b, y_b, w_b, h_b)$ be parameterized in $\mathbb{R}^4$. 
We embed each bounding box into a 2D Gaussian distribution $\mathcal{N}(\mu, \Sigma)$ with:
$$\mu = [x, y]^T, \quad \Sigma = \text{diag}\left(\frac{w^2}{4}, \frac{h^2}{4}\right)$$

The 2-Wasserstein distance $\mathcal{W}_2(\mathcal{N}_a, \mathcal{N}_b)$ between these Gaussian distributions in closed form is:
$$\mathcal{W}_2^2(\mathcal{N}_a, \mathcal{N}_b) = (x_a - x_b)^2 + (y_a - y_b)^2 + \frac{(w_a - w_b)^2 + (h_a - h_b)^2}{4}$$

To enforce scale-invariance, we define the normalized squared Wasserstein distance:
$$\mathcal{D}_{\mathcal{W}}^2(A, B) = \frac{(x_a - x_b)^2}{\bar{w}_{ab}^2} + \frac{(y_a - y_b)^2}{\bar{h}_{ab}^2} + \ln^2\left(\frac{w_a}{w_b}\right) + \ln^2\left(\frac{h_a}{h_b}\right)$$
where $\bar{w}_{ab}^2 = \frac{w_a^2 + w_b^2}{2}$ and $\bar{h}_{ab}^2 = \frac{h_a^2 + h_b^2}{2}$.

## 2. The Scale-Homotopy Operator
We define the characteristic geometric scale $s(B) = \sqrt{w_b \cdot h_b}$ and characteristic scale threshold $\sigma_0 > 0$ (default $\sigma_0 = 8.0\text{px}$).
The continuous Homotopy parameter $\gamma: \mathbb{R}_{>0} \to (0, 1)$ is:
$$\gamma(s) = \frac{s^2}{s^2 + \sigma_0^2}$$

The **Homotopy Wasserstein-IoU Similarity** $\mathcal{S}_{\text{H-WIoU}}$ is formulated as the multiplicative homotopy manifold:
$$\mathcal{S}_{\text{H-WIoU}}(A, B) = [\text{IoU}(A, B)]^{\gamma(s_B)} \cdot \exp\left(-(1 - \gamma(s_B))\mathcal{D}_{\mathcal{W}}^2(A, B)\right)$$

## 3. Boundary & Asymptotic Properties
* **Proposition 1 (Regularity & Limits)**:
  1. Monotonicity: $\frac{d\gamma}{ds} = \frac{2s\sigma_0^2}{(s^2 + \sigma_0^2)^2} > 0$ for all $s > 0$.
  2. Microscopic Limit ($s \to 0^+$):
     $$\lim_{s \to 0^+} \gamma(s) = 0 \implies \lim_{s \to 0^+} \mathcal{S}_{\text{H-WIoU}}(A, B) = \exp\left(-\mathcal{D}_{\mathcal{W}}^2(A, B)\right)$$
  3. Macroscopic Limit ($s \to \infty$):
     $$\lim_{s \to \infty} \gamma(s) = 1 \implies \lim_{s \to \infty} \mathcal{S}_{\text{H-WIoU}}(A, B) = \text{IoU}(A, B)$$
