---
title: "Paper A SA-ALW Conference Refinement Plan"
type: source
tags: [sa-alw, paper-a, conference, protocol, reproducibility]
date: 2026-08-02
source_file: raw/Paper_A_SA_ALW_Conference_Refinement_Plan.md
---

## Summary

This source freezes a conference-refinement program for one focused Paper A:
ALW is the canonical base formulation and SA-ALW is the sole proposed method.
It replaces historical result reuse with phased gates for method correctness,
source-disjoint data, original-image evaluation, public benchmarks, matched
multi-seed experiments, and claim-to-artifact traceability.

## Key Claims

- Paper A excludes CBL, ICBL, routing, refinement, P2, SAC, HFP/SDP, SAH-GD,
  and later performance-research branches.
- Historical tile-level and reused-test results are diagnostic only.
- Canonical ALW uses per-axis mean-square position denominators and squared
  log-ratio shape terms.
- SA-ALW is target-conditioned and adds no learnable parameters, but it is not
  symmetric or jointly scale-invariant under absolute-scale schedules.
- Main evidence requires original-image evaluation, public TinyPerson and
  AI-TOD-v2 distributions, matched seeds `42/123/2024`, and uncertainty for the
  primary comparison.

## Key Decisions

- `paper_a/` is the only submission-facing workspace.
- Standard original-image COCO AP is the cross-dataset primary outcome;
  benchmark-official metrics are reported with distinct labels.
- The final test stays closed until code, data, config, checkpoint selection,
  fusion, evaluator, and claims are frozen.
- A failed gate produces `REVISE` or `NO-GO`; it cannot be repaired by stronger
  prose.

## Connections

- [[Anisotropic Log-Wasserstein Distance (ALW)]] - canonical base formulation.
- [[Scale-Adaptive Anisotropic Log-Wasserstein Distance (SA-ALW)]] - proposed
  target-conditioned extension.
- [[SA-ALW Paper Refinement Phase 0-2 - 2026-08-02]] - current execution and
  gate results.
- [[Maximum-Performance Research Checkpoint - 2026-08-02]] - separate research
  handoff whose CBL-family results are outside Paper A.

## Contradictions

- Supersedes the old [[Paper Rewrite Summary - 2026-07-06]] implication that
  paper writing and numerical verification were complete. Those manuscripts
  used tile-level/reused-test evidence and non-canonical ALW/SA-ALW code paths.
- Corrects older wiki descriptions of SA-ALW schedules as learned or already
  frozen at `5.6/28.7`; Paper A requires explicit non-learned schedules fitted
  only on the eventual frozen training split.

