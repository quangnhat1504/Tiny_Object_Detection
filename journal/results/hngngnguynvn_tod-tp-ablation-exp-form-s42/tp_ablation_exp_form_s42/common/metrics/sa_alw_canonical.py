"""Canonical ALW and SA-ALW geometry for Paper A.

Legacy metric names intentionally remain in ``alw.py`` and ``sa_alw.py`` so
historical checkpoints can still be reconstructed. This module implements the
paper contract directly: squared log-ratio shape terms, clipped target-scale
schedules, pairwise similarities for assignment, and aligned distances for
regression.
"""
from __future__ import annotations

import math

import torch


EPS = 1e-6


def validate_schedule(
    *,
    s_min: float,
    s_max: float,
    beta_min: float,
    beta_max: float,
    w_min: float,
    w_max: float,
    schedule_form: str = "linear",
) -> None:
    values = (s_min, s_max, beta_min, beta_max, w_min, w_max)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("SA-ALW schedule values must be finite")
    if not 0.0 < s_min < s_max:
        raise ValueError("SA-ALW requires 0 < s_min < s_max")
    if not 0.0 < beta_min <= beta_max:
        raise ValueError("SA-ALW requires 0 < beta_min <= beta_max")
    if not 0.0 < w_min <= w_max:
        raise ValueError("SA-ALW requires 0 < w_min <= w_max")
    if schedule_form not in {"linear", "log_linear"}:
        raise ValueError("SA-ALW schedule_form must be linear or log_linear")


def scale_interpolation(
    gt_wh: torch.Tensor,
    *,
    s_min: float,
    s_max: float,
    schedule_form: str = "linear",
) -> torch.Tensor:
    """Return clipped small-object emphasis ``u(s)`` in ``[0, 1]``."""
    if gt_wh.ndim != 2 or gt_wh.shape[-1] != 2:
        raise ValueError("gt_wh must have shape [N, 2]")
    if not 0.0 < float(s_min) < float(s_max):
        raise ValueError("SA-ALW requires 0 < s_min < s_max")
    if schedule_form not in {"linear", "log_linear"}:
        raise ValueError("SA-ALW schedule_form must be linear or log_linear")
    safe_wh = gt_wh.clamp_min(EPS)
    scale = torch.sqrt(safe_wh[:, 0] * safe_wh[:, 1])
    if schedule_form == "linear":
        interpolation = (s_max - scale) / (s_max - s_min)
    else:
        interpolation = (
            math.log(s_max) - torch.log(scale.clamp_min(EPS))
        ) / (math.log(s_max) - math.log(s_min))
    return interpolation.clamp(0.0, 1.0)


def scale_adaptive_beta(
    gt_wh: torch.Tensor,
    *,
    s_min: float,
    s_max: float,
    beta_min: float,
    beta_max: float,
    schedule_form: str = "linear",
) -> torch.Tensor:
    if not 0.0 < float(beta_min) <= float(beta_max):
        raise ValueError("SA-ALW requires 0 < beta_min <= beta_max")
    u = scale_interpolation(
        gt_wh, s_min=s_min, s_max=s_max, schedule_form=schedule_form
    )
    return beta_min + (beta_max - beta_min) * u


def scale_adaptive_position_weight(
    gt_wh: torch.Tensor,
    *,
    s_min: float,
    s_max: float,
    w_min: float,
    w_max: float,
    schedule_form: str = "linear",
) -> torch.Tensor:
    if not 0.0 < float(w_min) <= float(w_max):
        raise ValueError("SA-ALW requires 0 < w_min <= w_max")
    u = scale_interpolation(
        gt_wh, s_min=s_min, s_max=s_max, schedule_form=schedule_form
    )
    return w_min + (w_max - w_min) * u


def _safe_sqrt(value: torch.Tensor, epsilon: float = EPS) -> torch.Tensor:
    """Zero-preserving square root with a finite derivative at identity."""
    return torch.sqrt(value.clamp_min(0.0) + epsilon * epsilon) - epsilon


def _pairwise_alw_squared(
    xn: torch.Tensor,
    yn: torch.Tensor,
    wn: torch.Tensor,
    hn: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    wg: torch.Tensor,
    hg: torch.Tensor,
    *,
    position_weight: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    wn = wn.clamp_min(EPS)
    hn = hn.clamp_min(EPS)
    wg = wg.clamp_min(EPS)
    hg = hg.clamp_min(EPS)

    dx = xn[:, None] - xg[None, :]
    dy = yn[:, None] - yg[None, :]
    sx = (wn[:, None].square() + wg[None, :].square()) / 2.0
    sy = (hn[:, None].square() + hg[None, :].square()) / 2.0
    position = dx.square() / sx.clamp_min(EPS) + dy.square() / sy.clamp_min(EPS)
    log_w = torch.log(wn)[:, None] - torch.log(wg)[None, :]
    log_h = torch.log(hn)[:, None] - torch.log(hg)[None, :]
    shape = log_w.square() + log_h.square()
    return (position_weight * position + shape).clamp_min(0.0)


def _aligned_alw_squared(
    xn: torch.Tensor,
    yn: torch.Tensor,
    wn: torch.Tensor,
    hn: torch.Tensor,
    xg: torch.Tensor,
    yg: torch.Tensor,
    wg: torch.Tensor,
    hg: torch.Tensor,
    *,
    position_weight: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    inputs = (xn, yn, wn, hn, xg, yg, wg, hg)
    if any(tensor.ndim != 1 for tensor in inputs):
        raise ValueError("aligned ALW inputs must be one-dimensional")
    if len({tensor.shape[0] for tensor in inputs}) != 1:
        raise ValueError("aligned ALW inputs must have equal length")

    wn = wn.clamp_min(EPS)
    hn = hn.clamp_min(EPS)
    wg = wg.clamp_min(EPS)
    hg = hg.clamp_min(EPS)
    sx = (wn.square() + wg.square()) / 2.0
    sy = (hn.square() + hg.square()) / 2.0
    position = (xn - xg).square() / sx.clamp_min(EPS)
    position = position + (yn - yg).square() / sy.clamp_min(EPS)
    shape = (torch.log(wn) - torch.log(wg)).square()
    shape = shape + (torch.log(hn) - torch.log(hg)).square()
    return (position_weight * position + shape).clamp_min(0.0)


def pairwise_alw_distance_squared(xn, yn, wn, hn, xg, yg, wg, hg, **_) -> torch.Tensor:
    return _pairwise_alw_squared(xn, yn, wn, hn, xg, yg, wg, hg)


def aligned_alw_distance_squared(xn, yn, wn, hn, xg, yg, wg, hg, **_) -> torch.Tensor:
    return _aligned_alw_squared(xn, yn, wn, hn, xg, yg, wg, hg)


def aligned_alw_distance(xn, yn, wn, hn, xg, yg, wg, hg, **_) -> torch.Tensor:
    return _safe_sqrt(
        aligned_alw_distance_squared(xn, yn, wn, hn, xg, yg, wg, hg)
    )


def compute_alw_similarity(
    xn,
    yn,
    wn,
    hn,
    xg,
    yg,
    wg,
    hg,
    *,
    beta: float = 8.0,
    **_,
) -> torch.Tensor:
    if not math.isfinite(float(beta)) or beta <= 0:
        raise ValueError("ALW beta must be finite and positive")
    distance = _safe_sqrt(
        pairwise_alw_distance_squared(xn, yn, wn, hn, xg, yg, wg, hg)
    )
    return torch.exp(-beta * distance)


def _sa_position_weight(
    wg: torch.Tensor,
    hg: torch.Tensor,
    *,
    use_sa_pos_weight: bool,
    s_min: float,
    s_max: float,
    w_min: float,
    w_max: float,
    schedule_form: str,
) -> torch.Tensor | float:
    if not use_sa_pos_weight:
        return 1.0
    gt_wh = torch.stack((wg, hg), dim=1)
    return scale_adaptive_position_weight(
        gt_wh,
        s_min=s_min,
        s_max=s_max,
        w_min=w_min,
        w_max=w_max,
        schedule_form=schedule_form,
    )


def pairwise_sa_alw_distance_squared(
    xn,
    yn,
    wn,
    hn,
    xg,
    yg,
    wg,
    hg,
    *,
    s_min: float,
    s_max: float,
    w_min: float,
    w_max: float,
    use_sa_pos_weight: bool = True,
    schedule_form: str = "linear",
    **_,
) -> torch.Tensor:
    weight = _sa_position_weight(
        wg,
        hg,
        use_sa_pos_weight=use_sa_pos_weight,
        s_min=s_min,
        s_max=s_max,
        w_min=w_min,
        w_max=w_max,
        schedule_form=schedule_form,
    )
    if isinstance(weight, torch.Tensor):
        weight = weight[None, :]
    return _pairwise_alw_squared(
        xn, yn, wn, hn, xg, yg, wg, hg, position_weight=weight
    )


def aligned_sa_alw_distance_squared(
    xn,
    yn,
    wn,
    hn,
    xg,
    yg,
    wg,
    hg,
    *,
    s_min: float,
    s_max: float,
    w_min: float,
    w_max: float,
    use_sa_pos_weight: bool = True,
    schedule_form: str = "linear",
    **_,
) -> torch.Tensor:
    weight = _sa_position_weight(
        wg,
        hg,
        use_sa_pos_weight=use_sa_pos_weight,
        s_min=s_min,
        s_max=s_max,
        w_min=w_min,
        w_max=w_max,
        schedule_form=schedule_form,
    )
    return _aligned_alw_squared(
        xn, yn, wn, hn, xg, yg, wg, hg, position_weight=weight
    )


def aligned_sa_alw_distance(xn, yn, wn, hn, xg, yg, wg, hg, **kwargs) -> torch.Tensor:
    return _safe_sqrt(
        aligned_sa_alw_distance_squared(
            xn, yn, wn, hn, xg, yg, wg, hg, **kwargs
        )
    )


def compute_sa_alw_similarity(
    xn,
    yn,
    wn,
    hn,
    xg,
    yg,
    wg,
    hg,
    *,
    s_min: float,
    s_max: float,
    beta_min: float,
    beta_max: float,
    w_min: float,
    w_max: float,
    beta: float = 8.0,
    use_sa_beta: bool = True,
    use_sa_pos_weight: bool = True,
    schedule_form: str = "linear",
    **_,
) -> torch.Tensor:
    validate_schedule(
        s_min=s_min,
        s_max=s_max,
        beta_min=beta_min,
        beta_max=beta_max,
        w_min=w_min,
        w_max=w_max,
        schedule_form=schedule_form,
    )
    squared = pairwise_sa_alw_distance_squared(
        xn,
        yn,
        wn,
        hn,
        xg,
        yg,
        wg,
        hg,
        s_min=s_min,
        s_max=s_max,
        w_min=w_min,
        w_max=w_max,
        use_sa_pos_weight=use_sa_pos_weight,
        schedule_form=schedule_form,
    )
    distance = _safe_sqrt(squared)
    if use_sa_beta:
        target_beta = scale_adaptive_beta(
            torch.stack((wg, hg), dim=1),
            s_min=s_min,
            s_max=s_max,
            beta_min=beta_min,
            beta_max=beta_max,
            schedule_form=schedule_form,
        )[None, :]
    else:
        if not math.isfinite(float(beta)) or beta <= 0:
            raise ValueError("ALW beta must be finite and positive")
        target_beta = beta
    return torch.exp(-target_beta * distance)


def compute_sa_alw_beta_only_similarity(*args, **kwargs) -> torch.Tensor:
    kwargs["use_sa_beta"] = True
    kwargs["use_sa_pos_weight"] = False
    return compute_sa_alw_similarity(*args, **kwargs)


def compute_sa_alw_pos_only_similarity(*args, **kwargs) -> torch.Tensor:
    kwargs["use_sa_beta"] = False
    kwargs["use_sa_pos_weight"] = True
    return compute_sa_alw_similarity(*args, **kwargs)


name = "sa_alw_canonical"
needs_reliability_thr = False
