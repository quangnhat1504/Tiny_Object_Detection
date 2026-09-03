"""Metric registry — single source of truth.

Each metric module exposes:
  - compute_rfd(xn, yn, wn, hn, xg, yg, wg, hg, **kwargs) -> Tensor [N, M]
       Similarity matrix in (0, 1].
  - name (str)
  - needs_reliability_thr (bool)

For ablation variants, multiple ALW variants live in alw.py.
"""
from __future__ import annotations
from functools import partial
from typing import Callable, Dict, Optional

from . import alw, igwd, iou, nwd, sa_alw, sa_alw_canonical, h_wiou, dynamic_uncertainty_h_wiou, wavelet_h_wiou, oriented_h_wiou, cascade_homotopy, entropy_homotopy

METRIC_REGISTRY: Dict[str, Callable] = {
    # Baselines (Phase 1)
    "standard":        iou.compute_rfd,
    "iou":             iou.compute_rfd,
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
    # Paper A canonical methods. Legacy names above remain checkpoint-stable.
    "alw_canonical": sa_alw_canonical.compute_alw_similarity,
    "sa_alw_canonical": sa_alw_canonical.compute_sa_alw_similarity,
    "sa_alw_canonical_beta_only": (
        sa_alw_canonical.compute_sa_alw_beta_only_similarity),
    "sa_alw_canonical_pos_only": (
        sa_alw_canonical.compute_sa_alw_pos_only_similarity),
    # Homotopy Wasserstein-IoU (Unified Continuous Homotopy)
    "h_wiou": h_wiou.compute_h_wiou_similarity,
    "du_hwiou": h_wiou.compute_h_wiou_similarity,
    "sw_hwiou": h_wiou.compute_h_wiou_similarity,
    "oriented_h_wiou": oriented_h_wiou.oriented_h_wiou_similarity,
    "eh_wiou": entropy_homotopy.compute_entropy_homotopy_similarity,
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
    "alw_canonical": "ALW (canonical)",
    "sa_alw_canonical": "SA-ALW (canonical)",
    "sa_alw_canonical_beta_only": "SA-ALW (canonical beta only)",
    "sa_alw_canonical_pos_only": "SA-ALW (canonical position only)",
    "h_wiou": "H-WIoU (Homotopy Wasserstein-IoU)",
    "igwd_log_shape":       "IGWD (+ log-shape)",
    "igwd_anisotropic_s":   "IGWD (+ anisotropic-S)",
}

# Metrics that need an adaptive reliability_thr (data-dependent)
NEEDS_RELIABILITY = {"alw_full", "alw_reliability_only", "alw_charbonnier_only",
                     "igwd_with_reliability",
                     "sa_alw_full", "sa_alw_beta_only", "sa_alw_pos_only"}

CANONICAL_METRICS = {
    "alw_canonical",
    "sa_alw_canonical",
    "sa_alw_canonical_beta_only",
    "sa_alw_canonical_pos_only",
}
CANONICAL_SA_METRICS = CANONICAL_METRICS - {"alw_canonical"}


def get_metric_fn(name: str) -> Callable:
    if name not in METRIC_REGISTRY:
        raise ValueError(
            f"Unknown metric: {name}. "
            f"Available: {list(METRIC_REGISTRY.keys())}"
        )
    return METRIC_REGISTRY[name]


def get_metric_distance_fn(name: str) -> Optional[Callable]:
    """Return an aligned regression distance for canonical Paper A methods."""
    if name == "alw_canonical":
        return sa_alw_canonical.aligned_alw_distance
    if name == "sa_alw_canonical_beta_only":
        return sa_alw_canonical.aligned_alw_distance
    if name in {"sa_alw_canonical", "sa_alw_canonical_pos_only"}:
        return sa_alw_canonical.aligned_sa_alw_distance
    if name in {"h_wiou", "du_hwiou", "sw_hwiou"}:
        return h_wiou.aligned_h_wiou_loss
    if name == "eh_wiou":
        return entropy_homotopy.aligned_entropy_homotopy_loss
    if name == "oriented_h_wiou":
        return oriented_h_wiou.aligned_oriented_h_wiou_loss
    return None


def configure_metric(
    name: str,
    *,
    beta: float = 8.0,
    s_min: float | None = None,
    s_max: float | None = None,
    beta_min: float | None = None,
    beta_max: float | None = None,
    w_min: float | None = None,
    w_max: float | None = None,
    schedule_form: str = "linear",
    h_wiou_sigma_0: float = 8.0,
    h_wiou_form: str = "rational",
    h_wiou_static_gamma: float = 0.5,
    h_wiou_sigmoid_tau: float = 2.0,
) -> tuple[Callable, Optional[Callable], dict]:
    """Build assignment and aligned-regression functions plus run metadata."""
    similarity_fn = get_metric_fn(name)
    distance_fn = get_metric_distance_fn(name)
    metadata = {"canonical": name in CANONICAL_METRICS, "beta": float(beta)}
    if name in {"h_wiou", "du_hwiou", "sw_hwiou"}:
        similarity_fn = partial(
            similarity_fn,
            sigma_0=float(h_wiou_sigma_0),
            form=str(h_wiou_form),
            static_gamma=float(h_wiou_static_gamma),
            sigmoid_tau=float(h_wiou_sigmoid_tau),
        )
        if distance_fn is not None:
            distance_fn = partial(
                distance_fn,
                sigma_0=float(h_wiou_sigma_0),
                form=str(h_wiou_form),
                static_gamma=float(h_wiou_static_gamma),
                sigmoid_tau=float(h_wiou_sigmoid_tau),
            )
        metadata["h_wiou_sigma_0"] = float(h_wiou_sigma_0)
        metadata["h_wiou_form"] = str(h_wiou_form)
        return similarity_fn, distance_fn, metadata
    if name == "eh_wiou":
        similarity_fn = partial(
            similarity_fn,
            sigma_0=float(h_wiou_sigma_0),
            beta=float(h_wiou_static_gamma),
        )
        if distance_fn is not None:
            distance_fn = partial(
                distance_fn,
                sigma_0=float(h_wiou_sigma_0),
                beta=float(h_wiou_static_gamma),
            )
        metadata["h_wiou_sigma_0"] = float(h_wiou_sigma_0)
        metadata["h_wiou_form"] = "eh_wiou"
        return similarity_fn, distance_fn, metadata
    if name == "alw_canonical":
        return partial(similarity_fn, beta=beta), distance_fn, metadata
    if name not in CANONICAL_SA_METRICS:
        return similarity_fn, distance_fn, metadata

    schedule = {
        "s_min": s_min,
        "s_max": s_max,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "w_min": w_min,
        "w_max": w_max,
    }
    missing = [key for key, value in schedule.items() if value is None]
    if missing:
        raise ValueError(
            "Canonical SA-ALW requires an explicit frozen train-derived schedule; missing: "
            + ", ".join(sorted(missing))
        )
    typed_schedule = {key: float(value) for key, value in schedule.items()}
    sa_alw_canonical.validate_schedule(
        **typed_schedule, schedule_form=schedule_form
    )
    use_sa_beta = name != "sa_alw_canonical_pos_only"
    use_sa_pos_weight = name != "sa_alw_canonical_beta_only"
    shared = {
        **typed_schedule,
        "beta": float(beta),
        "use_sa_beta": use_sa_beta,
        "use_sa_pos_weight": use_sa_pos_weight,
        "schedule_form": schedule_form,
    }
    similarity_fn = partial(similarity_fn, **shared)
    if name == "sa_alw_canonical_beta_only":
        distance_fn = sa_alw_canonical.aligned_alw_distance
    else:
        distance_fn = partial(
            distance_fn,
            s_min=typed_schedule["s_min"],
            s_max=typed_schedule["s_max"],
            w_min=typed_schedule["w_min"],
            w_max=typed_schedule["w_max"],
            use_sa_pos_weight=use_sa_pos_weight,
            schedule_form=schedule_form,
        )
    metadata.update(typed_schedule)
    metadata.update(
        {
            "use_sa_beta": use_sa_beta,
            "use_sa_pos_weight": use_sa_pos_weight,
            "schedule_form": schedule_form,
            "schedule_source": "explicit_frozen_train_config",
        }
    )
    return similarity_fn, distance_fn, metadata
