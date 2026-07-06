"""Metric registry — single source of truth.

Each metric module exposes:
  - compute_rfd(xn, yn, wn, hn, xg, yg, wg, hg, **kwargs) -> Tensor [N, M]
       Similarity matrix in (0, 1].
  - name (str)
  - needs_reliability_thr (bool)

For ablation variants, multiple ALW variants live in alw.py.
"""
from __future__ import annotations
from typing import Callable, Dict

from . import iou, nwd, igwd, alw, sa_alw

METRIC_REGISTRY: Dict[str, Callable] = {
    # Baselines (Phase 1)
    "ciou":            iou.compute_rfd,
    "nwd":             nwd.compute_rfd,
    "igwd":            igwd.compute_rfd,
    "alw_full":        alw.compute_rfd,            # default = full
    # Placement variants (Phase 2)
    # Note: same metric module, placement controlled by caller
    # Component ablation (Phase 3)
    "alw_aniso_only":      alw.compute_aniso_only,
    "alw_original":        alw.compute_aniso_only,   # alias: original ALW (aniso+log, no R, no Charbonnier)
    "alw_reliability_only": alw.compute_reliability_only,
    "alw_charbonnier_only": alw.compute_charbonnier_only,
    "igwd_with_reliability": igwd.compute_with_reliability,
    # SA-ALW (Phase 2.7-2.8)
    "sa_alw_full":       sa_alw.compute_rfd,          # SA-beta + SA-pos-weight
    "sa_alw_beta_only":  sa_alw.compute_sa_beta_only, # SA-beta only
    "sa_alw_pos_only":   sa_alw.compute_sa_pos_only,  # SA-pos-weight only
    # IGWD ablations (Phase 2.4-2.5) — fix cấu trúc của IGWD hướng tới ALW
    "igwd_log_shape":      igwd.compute_log_shape,       # isotropic + log-ratio
    "igwd_anisotropic_s":  igwd.compute_anisotropic_s,   # anisotropic + Euclidean
}

METRIC_DISPLAY_NAME = {
    "ciou":              "CIoU (baseline)",
    "nwd":               "NWD",
    "igwd":              "IGWD",
    "alw_full":          "ALW (full)",
    "alw_aniso_only":    "ALW (aniso only)",
    "alw_original":      "ALW (original)",
    "alw_reliability_only": "ALW (+ reliability)",
    "alw_charbonnier_only": "ALW (+ Charbonnier)",
    "igwd_with_reliability": "IGWD (+ reliability)",
    "sa_alw_full":       "SA-ALW (full)",
    "sa_alw_beta_only":  "SA-ALW (SA-beta only)",
    "sa_alw_pos_only":      "SA-ALW (SA-pos only)",
    "igwd_log_shape":       "IGWD (+ log-shape)",
    "igwd_anisotropic_s":   "IGWD (+ anisotropic-S)",
}

# Metrics that need an adaptive reliability_thr (data-dependent)
NEEDS_RELIABILITY = {"alw_full", "alw_reliability_only", "alw_charbonnier_only",
                     "igwd_with_reliability",
                     "sa_alw_full", "sa_alw_beta_only", "sa_alw_pos_only"}


def get_metric_fn(name: str) -> Callable:
    if name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric: {name}. "
            f"Available: {list(METRIC_REGISTRY.keys())}"
        )
    return METRIC_REGISTRY[name]