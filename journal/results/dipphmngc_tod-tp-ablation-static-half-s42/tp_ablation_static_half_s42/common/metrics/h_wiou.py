"""
Homotopy Wasserstein-IoU (H-WIoU) Metric & Loss Module.

Constructs a continuous C^infinity homotopy between the Discrete Lebesgue Measure space (IoU)
and the Riemannian 2-Wasserstein Optimal Transport space (W_2):

    gamma(s) = s^2 / (s^2 + sigma_0^2)   where s = sqrt(w_gt * h_gt)
    S_HWIoU(A, B) = [IoU(A, B)]^gamma(s) * exp( - (1 - gamma(s)) * D_W^2(A, B) )
    L_HWIoU(A, B) = 1 - S_HWIoU(A, B)

Asymptotic Boundary Properties:
    lim_{s -> inf} gamma(s) = 1  ==> S_HWIoU -> IoU (Exact IoU precision for normal/tiny objects)
    lim_{s -> 0}   gamma(s) = 0  ==> S_HWIoU -> exp(-D_W^2) (Smooth Optimal Transport for micro objects)
"""
from __future__ import annotations
import math
import torch
from torchvision.ops import box_iou

EPS = 1e-6


def compute_scale_homotopy(
    gt_wh: torch.Tensor,
    sigma_0: float = 8.0,
    form: str = "rational",
    static_gamma: float = 0.5,
    sigmoid_tau: float = 2.0,
) -> torch.Tensor:
    """Compute continuous scale homotopy parameter gamma(s) in [0, 1].

    Args:
        gt_wh: Tensor [N, 2] of (width, height)
        sigma_0: Characteristic microscopic scale threshold (default: 8.0 px)
        form: Homotopy deformation function form:
              - 'rational': s^2 / (s^2 + sigma_0^2) (Standard)
              - 'exponential': 1 - exp(-s / sigma_0)
              - 'sigmoid': 1 / (1 + exp(-(s - sigma_0) / tau))
              - 'static': constant static_gamma
              - 'pure_w2': 0.0 (Pure Optimal Transport)
              - 'pure_iou': 1.0 (Pure Lebesgue Measure / IoU)
        static_gamma: Fixed gamma value when form='static' (default: 0.5)
        sigmoid_tau: Smoothness temperature for sigmoid form (default: 2.0)
    Returns:
        gamma: Tensor [N] in [0, 1]
    """
    if gt_wh.ndim != 2 or gt_wh.shape[-1] != 2:
        raise ValueError("gt_wh must have shape [N, 2]")

    if form == "pure_w2":
        return torch.zeros(gt_wh.shape[0], device=gt_wh.device, dtype=gt_wh.dtype)
    if form == "pure_iou":
        return torch.ones(gt_wh.shape[0], device=gt_wh.device, dtype=gt_wh.dtype)
    if form == "static":
        return torch.full((gt_wh.shape[0],), float(static_gamma), device=gt_wh.device, dtype=gt_wh.dtype).clamp(0.0, 1.0)

    safe_wh = gt_wh.clamp_min(EPS)
    s = torch.sqrt(safe_wh[:, 0] * safe_wh[:, 1])

    if form == "rational":
        s_sq = s.square()
        sigma_sq = float(sigma_0) ** 2
        gamma = s_sq / (s_sq + sigma_sq)
    elif form == "exponential":
        gamma = 1.0 - torch.exp(-s / max(float(sigma_0), EPS))
    elif form == "sigmoid":
        gamma = torch.sigmoid((s - float(sigma_0)) / max(float(sigmoid_tau), EPS))
    else:
        raise ValueError(f"Unknown homotopy form '{form}'; choices: rational, exponential, sigmoid, static, pure_w2, pure_iou")

    return gamma.clamp(0.0, 1.0)


def pairwise_wasserstein_distance_squared(
    xa: torch.Tensor,
    ya: torch.Tensor,
    wa: torch.Tensor,
    ha: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    wb: torch.Tensor,
    hb: torch.Tensor,
) -> torch.Tensor:
    """Compute pairwise scale-normalized 2-Wasserstein distance squared [N, M]."""
    wa = wa.clamp_min(EPS)
    ha = ha.clamp_min(EPS)
    wb = wb.clamp_min(EPS)
    hb = hb.clamp_min(EPS)

    dx = xa[:, None] - xb[None, :]
    dy = ya[:, None] - yb[None, :]
    sx = (wa[:, None].square() + wb[None, :].square()) / 2.0
    sy = (ha[:, None].square() + hb[None, :].square()) / 2.0

    position_term = (dx.square() / sx.clamp_min(EPS)) + (dy.square() / sy.clamp_min(EPS))
    log_w = torch.log(wa)[:, None] - torch.log(wb)[None, :]
    log_h = torch.log(ha)[:, None] - torch.log(hb)[None, :]
    shape_term = log_w.square() + log_h.square()

    return (position_term + shape_term).clamp_min(0.0)


def aligned_wasserstein_distance_squared(
    xa: torch.Tensor,
    ya: torch.Tensor,
    wa: torch.Tensor,
    ha: torch.Tensor,
    xb: torch.Tensor,
    yb: torch.Tensor,
    wb: torch.Tensor,
    hb: torch.Tensor,
) -> torch.Tensor:
    """Compute aligned scale-normalized 2-Wasserstein distance squared [N]."""
    wa = wa.clamp_min(EPS)
    ha = ha.clamp_min(EPS)
    wb = wb.clamp_min(EPS)
    hb = hb.clamp_min(EPS)

    dx = xa - xb
    dy = ya - yb
    sx = (wa.square() + wb.square()) / 2.0
    sy = (ha.square() + hb.square()) / 2.0

    position_term = (dx.square() / sx.clamp_min(EPS)) + (dy.square() / sy.clamp_min(EPS))
    log_w = torch.log(wa) - torch.log(wb)
    log_h = torch.log(ha) - torch.log(hb)
    shape_term = log_w.square() + log_h.square()

    return (position_term + shape_term).clamp_min(0.0)


def compute_h_wiou_similarity(
    xa: torch.Tensor,
    ya: torch.Tensor,
    wa: torch.Tensor,
    ha: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    wg: torch.Tensor,
    hg: torch.Tensor,
    *,
    sigma_0: float = 8.0,
    form: str = "rational",
    static_gamma: float = 0.5,
    sigmoid_tau: float = 2.0,
    **_,
) -> torch.Tensor:
    """Compute pairwise Homotopy Wasserstein-IoU similarity matrix [N, M].

    Args:
        xa, ya, wa, ha: Anchor boxes center and size [N]
        xg, yg, wg, hg: Ground truth boxes center and size [M]
        sigma_0: Homotopy scale inflection constant
        form: Homotopy deformation function form
        static_gamma: Fixed gamma value when form='static'
        sigmoid_tau: Smoothness temperature for sigmoid form
    Returns:
        similarity: Tensor [N, M] in [0, 1]
    """
    # 1. Compute pairwise standard IoU
    boxes_a = torch.stack([
        xa - wa / 2.0, ya - ha / 2.0,
        xa + wa / 2.0, ya + ha / 2.0
    ], dim=1)
    boxes_g = torch.stack([
        xg - wg / 2.0, yg - hg / 2.0,
        xg + wg / 2.0, yg + hg / 2.0
    ], dim=1)
    iou_mat = box_iou(boxes_a, boxes_g).clamp(min=EPS, max=1.0)  # [N, M]

    # 2. Compute pairwise Wasserstein distance squared
    dw_sq = pairwise_wasserstein_distance_squared(xa, ya, wa, ha, xg, yg, wg, hg)  # [N, M]

    # 3. Compute scale homotopy parameter gamma(s) for each GT box
    gt_wh = torch.stack([wg, hg], dim=1)
    gamma = compute_scale_homotopy(
        gt_wh, sigma_0=sigma_0, form=form, static_gamma=static_gamma, sigmoid_tau=sigmoid_tau
    )  # [M]
    gamma = gamma[None, :]  # broadcast to [1, M]

    # 4. Homotopy synthesis: S = (IoU)^gamma * exp(-(1-gamma) * D_W^2)
    iou_part = torch.pow(iou_mat, gamma)
    ot_part = torch.exp(-(1.0 - gamma) * dw_sq)

    sim = (iou_part * ot_part).clamp(0.0, 1.0)
    return sim


def aligned_h_wiou_loss(
    *args: torch.Tensor,
    sigma_0: float = 8.0,
    form: str = "rational",
    static_gamma: float = 0.5,
    sigmoid_tau: float = 2.0,
    **_,
) -> torch.Tensor:
    """Compute aligned Homotopy Wasserstein-IoU loss for bounding box regression.
    Supports both 2-tensor calling convention (pred_boxes [N,4], target_boxes [N,4])
    and 8-coordinate convention (xa, ya, wa, ha, xb, yb, wb, hb [N]).
    """
    if len(args) == 2:
        pred_boxes, target_boxes = args
        if pred_boxes.ndim != 2 or pred_boxes.shape[-1] != 4:
            raise ValueError("pred_boxes must have shape [N, 4]")
        if target_boxes.ndim != 2 or target_boxes.shape[-1] != 4:
            raise ValueError("target_boxes must have shape [N, 4]")

        xa = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
        ya = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0
        wa = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=EPS)
        ha = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=EPS)

        xb = (target_boxes[:, 0] + target_boxes[:, 2]) / 2.0
        yb = (target_boxes[:, 1] + target_boxes[:, 3]) / 2.0
        wb = (target_boxes[:, 2] - target_boxes[:, 0]).clamp(min=EPS)
        hb = (target_boxes[:, 3] - target_boxes[:, 1]).clamp(min=EPS)

        x1 = torch.max(pred_boxes[:, 0], target_boxes[:, 0])
        y1 = torch.max(pred_boxes[:, 1], target_boxes[:, 1])
        x2 = torch.min(pred_boxes[:, 2], target_boxes[:, 2])
        y2 = torch.min(pred_boxes[:, 3], target_boxes[:, 3])
        inter = (x2 - x1).clamp(min=0.0) * (y2 - y1).clamp(min=0.0)
        area_a = wa * ha
        area_b = wb * hb
        union = (area_a + area_b - inter).clamp_min(EPS)
        iou = (inter / union).clamp(min=EPS, max=1.0)
    elif len(args) == 8:
        xa, ya, wa, ha, xb, yb, wb, hb = args
        x1_a, y1_a, x2_a, y2_a = xa - wa / 2.0, ya - ha / 2.0, xa + wa / 2.0, ya + ha / 2.0
        x1_b, y1_b, x2_b, y2_b = xb - wb / 2.0, yb - hb / 2.0, xb + wb / 2.0, yb + hb / 2.0
        inter_w = (torch.min(x2_a, x2_b) - torch.max(x1_a, x1_b)).clamp(min=0.0)
        inter_h = (torch.min(y2_a, y2_b) - torch.max(y1_a, y1_b)).clamp(min=0.0)
        inter = inter_w * inter_h
        union = (wa * ha + wb * hb - inter).clamp_min(EPS)
        iou = (inter / union).clamp(min=EPS, max=1.0)
    else:
        raise ValueError(f"aligned_h_wiou_loss expects 2 or 8 arguments, got {len(args)}")

    # 1. Aligned Wasserstein Distance Squared
    dw_sq = aligned_wasserstein_distance_squared(xa, ya, wa, ha, xb, yb, wb, hb)

    # 2. Scale Homotopy Parameter gamma(s)
    target_wh = torch.stack([wb, hb], dim=1)
    gamma = compute_scale_homotopy(
        target_wh, sigma_0=sigma_0, form=form, static_gamma=static_gamma, sigmoid_tau=sigmoid_tau
    )

    # 3. Homotopy Similarity & Loss
    iou_part = torch.pow(iou, gamma)
    ot_part = torch.exp(-(1.0 - gamma) * dw_sq)
    similarity = (iou_part * ot_part).clamp(0.0, 1.0)

    loss = 1.0 - similarity
    return loss
