from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.metrics import sa_alw_canonical as canonical
from common.model import _hierarchical_assignment


TECHNICAL_SCHEDULE = {
    "s_min": 5.0,
    "s_max": 20.0,
    "beta_min": 8.0,
    "beta_max": 10.0,
    "w_min": 1.0,
    "w_max": 1.5,
    "schedule_form": "linear",
}
QUALITY_RATIO = 0.60
FIXED_BETA = 8.0


def _columns(boxes: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(boxes[:, index] for index in range(4))


def _candidate_grid(scale: float) -> torch.Tensor:
    rows = []
    for offset_ratio in (-0.50, -0.25, -0.10, 0.0, 0.10, 0.25, 0.50):
        for size_ratio in (0.75, 1.0, 1.25):
            rows.append(
                [offset_ratio * scale, 0.0, size_ratio * scale, scale]
            )
    return torch.tensor(rows, dtype=torch.float64)


def run_audit() -> dict[str, object]:
    ranking_rows = []
    for scale in (4.0, 8.0, 12.0, 20.0, 32.0):
        candidates = _candidate_grid(scale)
        target = torch.tensor([[0.0, 0.0, scale, scale]], dtype=torch.float64)
        fixed = canonical.compute_alw_similarity(
            *_columns(candidates), *_columns(target), beta=FIXED_BETA
        )[:, 0]
        adaptive = canonical.compute_sa_alw_beta_only_similarity(
            *_columns(candidates), *_columns(target), **TECHNICAL_SCHEDULE
        )[:, 0]
        fixed_order = torch.argsort(fixed, descending=True, stable=True)
        adaptive_order = torch.argsort(adaptive, descending=True, stable=True)
        ranking_rows.append(
            {
                "target_scale": scale,
                "target_beta": float(
                    canonical.scale_adaptive_beta(
                        target[:, 2:],
                        s_min=TECHNICAL_SCHEDULE["s_min"],
                        s_max=TECHNICAL_SCHEDULE["s_max"],
                        beta_min=TECHNICAL_SCHEDULE["beta_min"],
                        beta_max=TECHNICAL_SCHEDULE["beta_max"],
                    )[0]
                ),
                "rank_changes": int((fixed_order != adaptive_order).sum()),
                "top3_equal": bool(
                    torch.equal(fixed_order[:3], adaptive_order[:3])
                ),
            }
        )

    fixed_margin = -math.log(QUALITY_RATIO) / FIXED_BETA
    small_margin = -math.log(QUALITY_RATIO) / TECHNICAL_SCHEDULE["beta_max"]

    offsets = torch.tensor([0.0, 0.10, 0.15, 0.22, 0.30, 0.40])
    candidate_scale = torch.full_like(offsets, 4.0)
    zeros = torch.zeros_like(offsets)
    target_zero = torch.zeros(1)
    target_scale = torch.full((1,), 4.0)
    fixed_threshold_sim = canonical.compute_alw_similarity(
        offsets,
        zeros,
        candidate_scale,
        candidate_scale,
        target_zero,
        target_zero,
        target_scale,
        target_scale,
        beta=FIXED_BETA,
    )
    adaptive_threshold_sim = canonical.compute_sa_alw_beta_only_similarity(
        offsets,
        zeros,
        candidate_scale,
        candidate_scale,
        target_zero,
        target_zero,
        target_scale,
        target_scale,
        **TECHNICAL_SCHEDULE,
    )
    fixed_match = _hierarchical_assignment(
        fixed_threshold_sim,
        offsets,
        zeros,
        candidate_scale,
        candidate_scale,
        target_zero,
        target_zero,
        target_scale,
        target_scale,
        metric_fn=None,
    )
    adaptive_match = _hierarchical_assignment(
        adaptive_threshold_sim,
        offsets,
        zeros,
        candidate_scale,
        candidate_scale,
        target_zero,
        target_zero,
        target_scale,
        target_scale,
        metric_fn=None,
    )

    ownership_candidate = torch.tensor([[2.2, 0.0, 8.0, 8.0]])
    ownership_targets = torch.tensor(
        [[0.0, 0.0, 4.0, 4.0], [3.0, 0.0, 20.0, 20.0]]
    )
    fixed_ownership = canonical.compute_alw_similarity(
        *_columns(ownership_candidate),
        *_columns(ownership_targets),
        beta=FIXED_BETA,
    )[0]
    adaptive_ownership = canonical.compute_sa_alw_beta_only_similarity(
        *_columns(ownership_candidate),
        *_columns(ownership_targets),
        **TECHNICAL_SCHEDULE,
    )[0]

    target = torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float64)
    center_candidate = torch.tensor(
        [[math.sqrt(0.03 * 16.0), 0.0, 4.0, 4.0]], dtype=torch.float64
    )
    shape_candidate = torch.tensor(
        [[0.0, 0.0, 4.0 * math.exp(math.sqrt(0.04)), 4.0]],
        dtype=torch.float64,
    )
    rank_candidates = torch.cat((center_candidate, shape_candidate), dim=0)
    alw_squared = canonical.pairwise_alw_distance_squared(
        *_columns(rank_candidates), *_columns(target)
    )[:, 0]
    position_squared = canonical.pairwise_sa_alw_distance_squared(
        *_columns(rank_candidates),
        *_columns(target),
        s_min=TECHNICAL_SCHEDULE["s_min"],
        s_max=TECHNICAL_SCHEDULE["s_max"],
        w_min=TECHNICAL_SCHEDULE["w_min"],
        w_max=TECHNICAL_SCHEDULE["w_max"],
    )[:, 0]

    regression_prediction = torch.tensor(
        [[1.0, -0.5, 4.5, 3.8]], dtype=torch.float64
    )
    alw_regression = canonical.aligned_alw_distance(
        *_columns(regression_prediction), *_columns(target)
    )
    beta_only_regression = canonical.aligned_sa_alw_distance(
        *_columns(regression_prediction),
        *_columns(target),
        s_min=TECHNICAL_SCHEDULE["s_min"],
        s_max=TECHNICAL_SCHEDULE["s_max"],
        w_min=TECHNICAL_SCHEDULE["w_min"],
        w_max=TECHNICAL_SCHEDULE["w_max"],
        use_sa_pos_weight=False,
    )

    checks = {
        "beta_preserves_every_within_target_ranking": all(
            row["rank_changes"] == 0 and row["top3_equal"]
            for row in ranking_rows
        ),
        "beta_changes_quality_threshold_eligibility": int(
            (fixed_match >= 0).sum()
        )
        != int((adaptive_match >= 0).sum()),
        "beta_can_change_cross_target_owner": int(fixed_ownership.argmax())
        != int(adaptive_ownership.argmax()),
        "position_weight_can_reverse_center_shape_ranking": int(
            alw_squared.argmin()
        )
        != int(position_squared.argmin()),
        "beta_only_regression_is_exact_alw": bool(
            torch.equal(alw_regression, beta_only_regression)
        ),
    }

    return {
        "status": "PASS_TECHNICAL_PREFLIGHT" if all(checks.values()) else "FAIL",
        "evidence_class": "validation_evidence_not_submission_evidence",
        "schedule_use": "controlled_synthetic_probe_only_not_frozen_for_training",
        "schedule": TECHNICAL_SCHEDULE,
        "quality_ratio": QUALITY_RATIO,
        "within_target_ranking": ranking_rows,
        "quality_threshold": {
            "fixed_beta_distance_margin": fixed_margin,
            "small_target_beta_distance_margin": small_margin,
            "relative_margin_change": small_margin / fixed_margin - 1.0,
            "fixed_assigned_count": int((fixed_match >= 0).sum()),
            "adaptive_assigned_count": int((adaptive_match >= 0).sum()),
            "changed_candidate_offset": 0.22,
        },
        "cross_target_ownership": {
            "fixed_similarity": fixed_ownership.tolist(),
            "adaptive_similarity": adaptive_ownership.tolist(),
            "fixed_owner": int(fixed_ownership.argmax()),
            "adaptive_owner": int(adaptive_ownership.argmax()),
        },
        "position_ranking": {
            "candidate_order": ["center_error", "shape_error"],
            "alw_squared": alw_squared.tolist(),
            "adaptive_position_squared": position_squared.tolist(),
            "alw_winner": int(alw_squared.argmin()),
            "adaptive_position_winner": int(position_squared.argmin()),
        },
        "regression": {
            "alw_distance": float(alw_regression[0]),
            "beta_only_distance": float(beta_only_regression[0]),
        },
        "checks": checks,
        "decision": (
            "Do not credit beta with within-target ranking or regression effects. "
            "Retain beta-only and position-only controls in a preregistered "
            "validation pilot before selecting the full conference matrix."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper_a/diagnostics/saalw_mechanism_preflight.json"),
    )
    args = parser.parse_args()
    result = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASS_TECHNICAL_PREFLIGHT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
