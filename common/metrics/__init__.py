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

from . import iou, nwd, igwd, alw

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
}

# Metrics that need an adaptive reliability_thr (data-dependent)
NEEDS_RELIABILITY = {"alw_full", "alw_reliability_only", "alw_charbonnier_only",
                     "igwd_with_reliability"}


def get_metric_fn(name: str) -> Callable:
    if name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric: {name}. "
            f"Available: {list(METRIC_REGISTRY.keys())}"
        )
    return METRIC_REGISTRY[name]