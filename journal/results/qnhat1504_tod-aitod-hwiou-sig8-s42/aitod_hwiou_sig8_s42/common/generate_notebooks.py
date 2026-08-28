"""Generate all experiment notebooks from a single template.

Usage:
    python generate_notebooks.py

Produces notebooks in ../notebooks/ with names like:
  - 01_core_metric__ciou__everywhere__seed42.ipynb
  - 02_core_metric__nwd__la__seed42.ipynb
  - ...
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).parent
NB_DIR = HERE.parent / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)

TEMPLATE_PATH = HERE / "_notebook_template.py"
TEMPLATE_SOURCE = TEMPLATE_PATH.read_text()


# ═════════════════════════════════════════════════════════════════════════
# Experiment definitions (metric_name, placement, seed, phase, tag)
# ═════════════════════════════════════════════════════════════════════════
EXPERIMENTS: List[Dict] = []

# Phase 1 — Core metric comparison (4 metrics × 3 seeds = 12)
for metric in ["ciou", "nwd", "igwd", "alw_full"]:
    for seed in [42, 123, 2024]:
        placement = "everywhere" if metric == "ciou" else "la"
        EXPERIMENTS.append({
            "metric": metric, "placement": placement, "seed": seed,
            "phase": 1, "tag": "core_metric",
        })

# Phase 2 — Placement ablation for ALW (3 placements × 3 seeds = 9)
# Note: "la" placement already covered in Phase 1
for placement in ["la_loss", "la_loss_nms"]:
    for seed in [42, 123, 2024]:
        EXPERIMENTS.append({
            "metric": "alw_full", "placement": placement, "seed": seed,
            "phase": 2, "tag": "placement",
        })

# Phase 3 — Component ablation (5 new variants × 3 seeds = 15)
# Note: "alw_full" already in Phase 1
# "alw_original" is the baseline ALW formulation (aniso+log only, no R, no Charbonnier)
# Listed first as the reference point.
for variant in ["alw_original", "alw_aniso_only", "alw_reliability_only",
                "alw_charbonnier_only", "igwd_with_reliability"]:
    for seed in [42, 123, 2024]:
        EXPERIMENTS.append({
            "metric": variant, "placement": "la", "seed": seed,
            "phase": 3, "tag": "component",
        })


def make_notebook(metric: str, placement: str, seed: int) -> dict:
    """Build a notebook by substituting placeholders in template."""
    src = (TEMPLATE_SOURCE
           .replace("__METRIC__", metric)
           .replace("__PLACEMENT__", placement)
           .replace("__SEED__", str(seed)))
    src_lines = src.splitlines(keepends=True)

    cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "cell-0",
        "metadata": {},
        "outputs": [],
        "source": src_lines,
    }
    nb = {
        "cells": [cell],
        "metadata": {
            "kernelspec": {"display_name": "Python 3",
                            "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return nb


def main():
    # Clear old notebooks
    if NB_DIR.exists():
        for old in NB_DIR.glob("*.ipynb"):
            old.unlink()
            print(f"  removed {old.name}")

    print(f"\nGenerating {len(EXPERIMENTS)} notebooks in {NB_DIR}/")
    for exp in EXPERIMENTS:
        nb = make_notebook(exp["metric"], exp["placement"], exp["seed"])
        prefix = f"{exp['phase']:02d}_{exp['tag']}"
        fname = (f"{prefix}__{exp['metric']}__{exp['placement']}"
                 f"__seed{exp['seed']}.ipynb")
        path = NB_DIR / fname
        with open(path, "w") as f:
            json.dump(nb, f, indent=1)
        print(f"  ✓ {fname}")

    # Summary
    by_phase = {}
    for exp in EXPERIMENTS:
        by_phase.setdefault(exp["phase"], []).append(exp)
    print("\n=== Summary by phase ===")
    for p, exps in sorted(by_phase.items()):
        unique_metrics = set((e["metric"], e["placement"]) for e in exps)
        print(f"Phase {p}: {len(exps)} runs "
              f"({len(unique_metrics)} unique configs × "
              f"{len(exps) // max(len(unique_metrics), 1)} seeds)")


if __name__ == "__main__":
    main()