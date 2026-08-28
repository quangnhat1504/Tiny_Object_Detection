# Paper A G3 Pilot Decision Protocol

Status: `PREREGISTERED; TINYPERSON_SIDE_READY` (G1 TinyPerson bounds fitted;
PL-001 split frozen; pilot harness smoke-passed on all six methods;
account assigned `ngquangnht`; pre-run report and Kaggle dataset package
remaining)

## Purpose

The reduced-schedule TinyPerson validation pilot must answer two questions
before the multi-seed matrix consumes substantial GPU time:

1. Does any canonical SA-ALW variant have a positive validation signal over
   both the verified direct predecessor and pure canonical ALW?
2. Does the two-schedule full method outperform its beta-only and position-only
   components, or is one adaptive path the defensible method?

The second question is mandatory because the controlled mechanism preflight
proves that beta cannot alter within-target ranking and is absent from
regression. Its only assignment paths are threshold eligibility and cross-GT
ownership; position emphasis changes geometry in both assignment and
regression.

## Frozen Pilot Matrix

All six methods use seed `42`, the same reduced epoch/update budget, data order,
augmentation, detector/backbone, validation-COCO-AP checkpoint selector,
original-image evaluator, and fixed IoU-NMS:

The reference schedule is the linear P10/P90, beta `8 -> 10`, position-weight
`1 -> 1.5` configuration in `../schedules/endpoint_protocol.md`.

1. Standard IoU/Smooth-L1.
2. Verified direct predecessor.
3. Pure canonical ALW.
4. SA-ALW beta-only.
5. SA-ALW position-only.
6. Full SA-ALW.

For workload handoff, methods 1-3 and 6 form `PILOT-D1-S42`; the two component
controls form `PILOT-COMP-D1-S42`. The component shard must use the same account,
GPU request, hashes, and run budget as the main pilot shard. Both shards are one
scientific decision and must finish before G3 is called.

## Selection Rule

The primary selector is validation COCO AP from independently reloaded
checkpoints. AP50/AP75/size AP/AR are reported but do not replace the primary
selector.

- If full SA-ALW exceeds both components by more than `0.001` absolute AP,
  retain the two-schedule method.
- If position-only or beta-only exceeds full by more than `0.001` absolute AP,
  return to G0/G1 and revise the central method before any core run.
- Otherwise, among SA variants within `0.001` AP of the best SA result, prefer
  the simpler scheduled variant in this fixed order: position-only, beta-only,
  then full. This tie rule is frozen from the mechanism audit rather than chosen
  after observing metrics.
- A G3 `GO` requires the selected SA variant to exceed both the direct
  predecessor and canonical ALW in validation AP, with finite training and a
  nonzero audited mechanism effect.
- If a component is positive but endpoints appear mis-scaled, return `REVISE`
  to the bounded, preregistered validation sensitivity grid. Do not touch test.
- If no SA variant exceeds both references, return `NO-GO Paper A`; do not
  promote a best seed or legacy result.

This pilot selects a formulation for the matched three-seed matrix. It does not
enable a performance claim by itself.
