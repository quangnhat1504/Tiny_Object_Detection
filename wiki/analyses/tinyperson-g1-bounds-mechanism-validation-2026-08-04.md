---
title: TinyPerson G1 Bounds, Mechanism Diagnostics, and Validation Proposal - 2026-08-04
type: analysis
created: 2026-08-04
updated: 2026-08-04
sources: [Paper_A_SA_ALW_Conference_Refinement_Plan.md]
tags: [paper-a, tinyperson, schedules, mechanism, validation-split]
---

# TinyPerson G1 Bounds, Mechanism Diagnostics, and Validation Proposal - 2026-08-04

**Question:** What are TinyPerson's train-only scale bounds, does the SA-ALW assignment mechanism reproduce there, and how should its missing validation split be handled?

## Train-only scale bounds (fitted, cross-checked)

`schedules/tinyperson_train_p10_p90.json` (audit SHA-256 `2ae4fb56...`),
frozen transform `640/800`, train corner-task annotation SHA-256
`8474f124...`, 32,430 valid positives:

| Stat | TinyPerson | AI-TOD-v2 |
|------|-----------|-----------|
| P10 | **7.4328 px** | 6.1968 px |
| median | 15.7167 px | 9.4657 px |
| P90 | **44.8468 px** | 13.8564 px |
| max | 335.2195 px | 72.9712 px |

TinyPerson's target-scale distribution is far wider (beach surveillance
scenes plus web images). An independent recomputation reproduced both
percentiles and the audit hash. Bounds are candidates, not frozen run-config
endpoints.

## Anchor-assignment preflight reproduces the mechanism pattern

`audit_saalw_anchor_assignment.py` was generalized with
`--target-height/--target-width`; defaults reproduce the frozen AI-TOD-v2
result exactly on CUDA (parity check passed; only the `selection_rule` text
was clarified). TinyPerson crops resize to two orientations, so two seeded
(42) 64-image preflights ran:

- 800x640 (2,426 eligible): full SA-ALW changes 358 assignments vs ALW,
  positives 2,442 -> 2,360 (-3.36%), GT coverage identical 622/624.
- 640x800 (775 eligible): 493 changed assignments, positives
  3,532 -> 3,323 (-5.92%), coverage identical 944/944.

The threshold-dominated, coverage-preserving effect frozen for AI-TOD-v2
therefore reproduces on TinyPerson. Mechanism preflight
(`audit_saalw_mechanism.py`) is dataset-independent synthetic geometry, so it
needs no TinyPerson rerun.

## Validation-split proposal (PL-001, PROPOSED)

TinyPerson A1 material contains 37 video sources (6,832 crops) and 93
distinct web-image sources (1,424 crops). Frame/crop-level splits would leak
adjacent frames, so `paper_a/protocol_ledger.md` entry PL-001 proposes a
deterministic group-disjoint 20% split keyed by
`sha256("tinyperson_<video|image>_<identity>")`: validation = 7 videos + 19
image groups = 2,041 crops / 4,719 positives; train_sub = 6,215 crops /
27,711 positives. The entry is PROPOSED pending user freeze; see
[[TinyPerson Acquisition and Real-Data Fixture - 2026-08-04]] for access
classes.

## Status

`TINYPERSON_BOUNDS_FITTED_AND_CROSSCHECKED; MECHANISM_PREFLIGHT_PASS_BOTH_ORIENTATIONS; PL-001_PROPOSED; G1 CLOSER`
