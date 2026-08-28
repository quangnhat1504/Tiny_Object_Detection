# Paper A Placement Audit

Status: `CODE_PATH_AND_CUDA_SMOKE_VERIFIED; MECHANISM_QUANTIFICATION_PENDING`

## Call Graph

```text
scripts/train_frcnn_metric.py
  -> common.metrics.configure_metric(...)
       -> pairwise similarity callable
       -> aligned regression-distance callable
  -> common.model.build_model(...)
       -> MetricRPN.assign_targets_to_anchors(...)
            -> pairwise similarity
            -> _hierarchical_assignment(...)
       -> patched RoIHeads.forward(...)
            -> decode positive class-specific boxes
            -> _metric_aux_loss(...)
            -> aligned regression distance
       -> standard torchvision class-aware IoU-NMS
```

## Component Matrix

| Component | Assignment | Regression | NMS |
|---|---:|---:|---:|
| `D_ALW` / `D_SA` geometry | yes | yes | no |
| `beta(s)` | yes, through similarity | no | no |
| `w_pos(s)` | yes | yes | no |
| IoU-NMS | no | no | fixed |

## Placement Modes

| CLI placement | RPN assigner | RoI box loss | NMS | Paper A use |
|---|---|---|---|---|
| `la` | canonical pairwise similarity | standard torchvision loss | IoU | assignment-only ablation |
| `loss` | standard torchvision IoU matcher | canonical aligned distance | IoU | regression-only ablation |
| `la_loss` | canonical pairwise similarity | canonical aligned distance | IoU | full method |
| `la_loss_nms` | metric | metric | metric | rejected by Paper A guard |
| `saalw_assigner` | threshold/dynamic-threshold assigner | legacy metric path | IoU | outside Paper A core |

## Audit Findings

1. Legacy `alw_full` combines reliability gating and Charbonnier shaping and is
   not the canonical ALW denominator.
2. Legacy `alw_original` uses absolute log ratios in the shape sum rather than
   the squared log-ratio formulation frozen for Paper A.
3. Legacy SA-ALW always includes reliability and Charbonnier wrappers.
4. The legacy RoI metric loss averages an all-pairs similarity matrix and uses
   `1-K`; it does not implement the manuscript's aligned distance loss.
5. Canonical names use a separate aligned-distance callable, preserving legacy
   checkpoint reconstruction while preventing the all-pairs regression bug.
6. Canonical beta-only regression resolves to the exact ALW distance callable.
7. Schedule values are explicit and recorded; canonical SA-ALW cannot fall back
   to the unverified constants in `common/config.py`.
8. Canonical runs require validation COCO AP checkpoint selection; their frozen
   checkpoint artifact is `best_coco_ap.pt`, not legacy `best.pt`.
9. Controlled preflight confirms beta-only preserves within-GT ordering and has
   no regression path. Its only observed assignment paths are relative-threshold
   eligibility and cross-scale GT ownership; position emphasis can change the
   center-versus-shape ordering directly.

## Remaining Evidence

- Count assignment changes by scale and cause.
- Measure center/shape gradient norms by scale.
- Freeze schedule bounds only after the train split is repaired.

The synthetic preflight is recorded at
`diagnostics/saalw_mechanism_preflight.json`; it is technical evidence only.

## Technical Smoke Result

`diagnostics/canonical_detector_smoke_seed42.json` records a synthetic AMP
forward/backward step for each Paper A placement on an RTX 5070 Ti. All losses
are finite, the RPN/RoI activation flags match the requested placement, strict
joint-state reload has zero missing/unexpected keys, and every model has
`41,311,996` parameters, exactly matching the standard detector.
