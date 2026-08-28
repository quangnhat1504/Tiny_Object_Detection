"""
Oriented 2D Gaussian Homotopy Wasserstein-IoU (O-HWIoU) Module for Oriented Tiny Object Detection (AI-TOD-R).

Maps 5-parameter oriented bounding boxes (x, y, w, h, theta) to 2D Gaussian distributions
and constructs a scale homotopy between Rotated IoU and Riemannian 2-Wasserstein Distance:
    Sigma_i = R(theta_i) * diag(w_i^2/4, h_i^2/4) * R(theta_i)^T
    D_W2^2(N_1, N_2) = ||mu_1 - mu_2||_2^2 + Tr(Sigma_1 + Sigma_2 - 2 (Sigma_1^{1/2} Sigma_2 Sigma_1^{1/2})^{1/2})
    gamma(s) = s^2 / (s^2 + sigma_0^2)
    S_OHWIoU(A, B) = [RotatedIoU(A, B)]^gamma(s) * exp( - (1 - gamma(s)) * (D_W2^2 / sigma_0^2) )
"""
from __future__ import annotations
import math
import torch

EPS = 1e-6


def oriented_box_to_2d_gaussian(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    h: torch.Tensor,
    theta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert oriented boxes (x, y, w, h, theta) into 2D Gaussian covariance components.
    Args:
        x, y, w, h, theta: Tensors of shape [N] (theta in radians).
    Returns:
        sigma_xx, sigma_yy, sigma_xy: Covariance matrix components [N].
    """
    w_safe = w.clamp_min(EPS)
    h_safe = h.clamp_min(EPS)
    a = (w_safe / 2.0).square()
    b = (h_safe / 2.0).square()

    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    sigma_xx = a * cos_t.square() + b * sin_t.square()
    sigma_yy = a * sin_t.square() + b * cos_t.square()
    sigma_xy = (a - b) * cos_t * sin_t

    return sigma_xx, sigma_yy, sigma_xy


def oriented_wasserstein_distance_squared(
    x1: torch.Tensor,
    y1: torch.Tensor,
    w1: torch.Tensor,
    h1: torch.Tensor,
    theta1: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    w2: torch.Tensor,
    h2: torch.Tensor,
    theta2: torch.Tensor,
) -> torch.Tensor:
    """Compute aligned 2-Wasserstein distance squared between oriented 2D Gaussians [N]."""
    # 1. Center Euclidean distance
    center_dist_sq = (x1 - x2).square() + (y1 - y2).square()

    # 2. Covariance components
    sxx1, syy1, sxy1 = oriented_box_to_2d_gaussian(x1, y1, w1, h1, theta1)
    sxx2, syy2, sxy2 = oriented_box_to_2d_gaussian(x2, y2, w2, h2, theta2)

    tr_sigma = (sxx1 + syy1) + (sxx2 + syy2)

    # 3. Product trace approximation for 2x2 symmetric positive definite matrices:
    # Tr((Sigma_1^{1/2} Sigma_2 Sigma_1^{1/2})^{1/2}) = sqrt(Tr(Sigma_1 Sigma_2) + 2 sqrt(det(Sigma_1) det(Sigma_2)))
    det1 = (sxx1 * syy1 - sxy1.square()).clamp_min(EPS)
    det2 = (sxx2 * syy2 - sxy2.square()).clamp_min(EPS)
    tr_prod = sxx1 * sxx2 + syy1 * syy2 + 2.0 * sxy1 * sxy2

    cross_term = 2.0 * torch.sqrt((tr_prod + 2.0 * torch.sqrt(det1 * det2)).clamp_min(EPS))
    wasserstein_sq = center_dist_sq + (tr_sigma - cross_term).clamp_min(0.0)

    # Normalize by characteristic scale
    scale_norm = (w1 * h1 + w2 * h2) / 2.0
    return (wasserstein_sq / scale_norm.clamp_min(EPS)).clamp_min(0.0)


def oriented_h_wiou_similarity(
    x1: torch.Tensor,
    y1: torch.Tensor,
    w1: torch.Tensor,
    h1: torch.Tensor,
    theta1: torch.Tensor,
    x2: torch.Tensor,
    y2: torch.Tensor,
    w2: torch.Tensor,
    h2: torch.Tensor,
    theta2: torch.Tensor,
    approx_rotated_iou: torch.Tensor | None = None,
    sigma_0: float = 8.0,
) -> torch.Tensor:
    """Compute Oriented Homotopy Wasserstein-IoU similarity in [0, 1]."""
    dw_sq = oriented_wasserstein_distance_squared(
        x1, y1, w1, h1, theta1, x2, y2, w2, h2, theta2
    )

    s = torch.sqrt((w2.clamp_min(EPS)) * (h2.clamp_min(EPS)))
    gamma = (s.square() / (s.square() + float(sigma_0) ** 2)).clamp(0.0, 1.0)

    if approx_rotated_iou is None:
        # High-fidelity differentiable proxy for oriented IoU
        approx_rotated_iou = torch.exp(-dw_sq).clamp(min=EPS, max=1.0)

    iou_part = torch.pow(approx_rotated_iou.clamp(min=EPS, max=1.0), gamma)
    ot_part = torch.exp(-(1.0 - gamma) * dw_sq)
    return (iou_part * ot_part).clamp(0.0, 1.0)
