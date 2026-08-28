# Paper A Scope Contract

Version: `1.0`
Frozen: `2026-08-02`
Source: `raw/Paper_A_SA_ALW_Conference_Refinement_Plan.md`
Working title: `SA-ALW: Scale-Adaptive Anisotropic Log-Wasserstein Similarity for Tiny Object Detection`

## Research Question

Under a matched tiny-object detection protocol evaluated on original images,
does target-scale-conditioned ALW improve COCO AP over its direct predecessor
without adding learnable parameters, and what changes in assignment and
optimization explain any observed difference?

## Base Formulation

For positive-width and positive-height boxes, canonical ALW uses per-axis
mean-square position denominators and squared relative log-shape terms:

```text
S_x = (w_p^2 + w_t^2) / 2
S_y = (h_p^2 + h_t^2) / 2

D_ALW = (x_p-x_t)^2 / S_x + (y_p-y_t)^2 / S_y
        + log(w_p/w_t)^2 + log(h_p/h_t)^2
```

The base is a Wasserstein-inspired box distance/similarity formulation. Paper A
must not call it a mathematical metric unless all required metric axioms are
proved.

## Proposed Extension

Let target scale be `s = sqrt(w_t h_t)` and let both schedules be clipped to
the interval fitted from train data only:

```text
u(s) = clip((s_max - s) / (s_max - s_min), 0, 1)
beta(s) = beta_min + (beta_max - beta_min) * u(s)
w_pos(s) = w_min + (w_max - w_min) * u(s)

D_SA = w_pos(s) * [(x_p-x_t)^2 / S_x + (y_p-y_t)^2 / S_y]
       + log(w_p/w_t)^2 + log(h_p/h_t)^2
K_SA(p,t) = exp(-beta(s) * sqrt(D_SA))
```

This extension is target-conditioned. It is therefore not claimed to be
symmetric or jointly scale-invariant.

## Placement Contract

The intended full configuration applies the canonical similarity to label
assignment and the canonical distance to RoI box regression. Standard IoU-NMS
is fixed. Phase 1 must verify the actual call graph before this becomes a paper
claim.

The conference matrix must separately evaluate:

1. assignment only;
2. regression only;
3. assignment plus regression.

The beta schedule may be credited only to code paths in which beta appears.
Current expectation: beta affects similarity/assignment, not the distance-only
regression objective. This remains `PENDING_CODE_AUDIT`.

## In Scope

- Canonical ALW derivation and properties.
- SA-ALW target-scale-conditioned beta and position-weight schedules.
- Assignment, regression, and joint placement ablations.
- Original-image evaluation on public tiny-object benchmarks.
- Matched baselines, multi-seed uncertainty, mechanism diagnostics, efficiency,
  qualitative analysis, failure cases, limitations, ethics, and reproducibility.

## Out of Scope

- CBL and ICBL.
- Cascaded routing or cascaded detector refinement.
- P2, SAC, HFP, SDP, SAH-GD, PC-MR, PC-MOC, PC-MHFD, RA-TB, CR-SC-CBL, and
  every later performance-research branch.
- Metric-based NMS; IoU-NMS remains fixed.
- Tile-level metrics as paper outcomes.
- Reused test results as paper evidence.

Out-of-scope results remain in the research log and may not support Paper A
claims, even when their internal protocol is otherwise strong.

## Outcomes

Primary outcome:

- COCO AP on original images, computed by the official evaluator or a verified
  equivalent implementation.

Secondary outcomes:

- AP50, AP75, AP_S, AP_M, and AR100 on original images.
- AP_L only with ground-truth count and uncertainty; otherwise supplementary.
- Assignment-change rate and center/shape gradient norms by target-scale bin.
- Training time per epoch, end-to-end latency, peak VRAM, parameter count, and
  auditable compute overhead.

Checkpoint selection must use validation COCO AP because AP is primary. All
methods use the same schedule, seed set, data order where feasible, augmentation,
tiling, fusion, evaluator, and checkpoint rule.

## Dataset Boundary

- D1: official TinyPerson or its official protocol.
- D2: AI-TOD-v2.
- SOD-TinyPeopleInSea derivative: development or supplementary context only,
  after a group-disjoint split and original-image evaluator are verified.

All schedule percentiles and hyperparameters are derived from train only.
Validation is used for method and checkpoint selection. Final test remains
unopened until the full preregistration freeze is recorded.

## Seed Contract

Core comparisons use matched seeds `42`, `123`, and `2024`. A one-seed reduced
pilot is permitted on validation only. A single best seed cannot support the
primary performance claim.

## Evidence Status Vocabulary

- `diagnostic_invalid_protocol`: useful for debugging only; violates one or
  more submission protocol requirements.
- `diagnostic_valid_but_test_reused`: internally auditable but test access has
  already been consumed or reused.
- `validation_evidence`: eligible for method decisions, never a final test
  claim.
- `submission_evidence`: created only after G0-G3 and the final freeze.

No pre-refinement artifact is automatically promoted between statuses.

## Initial Claim Boundary

The authoritative row-level claim registry is `claims_ledger.csv`.

Potentially enableable after evidence:

- ALW uses per-axis anisotropic position normalization and squared log-ratio
  shape terms.
- Canonical ALW is non-negative, symmetric, dimensionless, and jointly
  scale-invariant for positive widths/heights before numerical clamping.
- SA-ALW is a target-conditioned extension with no additional learnable
  parameters.
- SA-ALW improves mean matched-seed original-image AP over the verified direct
  predecessor, only if the paired uncertainty supports it.

Forbidden at G0:

- first scale-adaptive Gaussian metric;
- parameter-free;
- state of the art;
- consistently superior;
- strict-localization improvement without AP75 uncertainty;
- detector-agnostic without a second audited detector;
- SA-ALW symmetry or scale invariance;
- any claim based on legacy tile-level or reused-test numbers.

## Freeze and Change Control

Changes to research question, method family, primary metric, datasets, seed set,
checkpoint selector, test access, or evidence vocabulary require a new version
of this contract and an entry in `paper_a/runs/manifest.jsonl`. Results already
observed may not be used to justify a retroactive protocol change.

