# Original-Image Evaluation Contract

Status: `RECONSTRUCTION_CORE_TESTED; BOTH_OFFICIAL_EVALUATOR_FIXTURES_TESTED`

## Unit of Evaluation

The evaluator receives one prediction dictionary and one ground-truth
dictionary per original image. Tiles are inference units only. A tile index may
never become a COCO image ID in the submission evaluator.

## Reconstruction Pipeline

1. Run the frozen detector independently on each tile.
2. Add the tile `(x, y)` offset to every predicted box.
3. Clip boxes to the original image bounds.
4. Drop invalid boxes and predictions below the frozen score floor.
5. Group by original image and class.
6. Apply the same fixed class-aware IoU-NMS rule to every method.
7. Keep the frozen `maxDets` count.
8. Build ground truth once from the original annotation, never from clipped tile
   annotations.
9. Call the official evaluator once over original images.

Implementation: `evaluation/tile_to_original.py`.

## Frozen Parameters

The following values remain `UNFROZEN` until validation-only calibration is
predeclared:

- score floor;
- tile size and overlap;
- class-aware IoU-NMS threshold;
- maxDets and official area ranges;
- ignore/crowd handling for each public benchmark.

No parameter may be selected on a final test.

## Benchmark Evaluators

Two labeled result families are required from the same reconstructed
original-image predictions:

1. `paper_primary_coco`: standard COCO AP over IoU `0.50:0.05:0.95`, with its
   exact `maxDets` and area settings frozen before the pilot.
2. `benchmark_official`: the pinned benchmark implementation and settings.

For TinyPerson, the official family uses the updated ignore-aware evaluator,
IoU thresholds `0.25/0.50/0.75`, `maxDets=200`, and its tiny size bins. For
AI-TOD-v2, it uses `cocoapi-aitod`, IoU `0.50:0.05:0.95`,
`maxDets=[1,100,1500]`, and the official very-tiny/tiny/small/medium bins.

The pinned AI-TOD-v2 evaluator reproduces AP, AP50, AP75, and tiny AP of `1.0`
on a perfect original-image fixture. The pinned TinyPerson evaluator reproduces
AP25, AP50, and AP75 of `1.0` while correctly ignoring a higher-scoring
detection inside an uncertain region through IOD. Both fixtures are mandatory
in the Kaggle runtime before any benchmark job.

The source commits and file hashes are frozen in
`official_evaluator_lock.json`. Values from these two families may not share an
unqualified metric label.

## Current Validation

`tests/test_tile_to_original.py` covers:

- offset inversion and clipping;
- overlapping-tile duplicate suppression;
- class-aware retention;
- empty original images;
- single-tile equivalence;
- maxDets enforcement;
- border-crossing objects;
- original GT count preservation.

`tests/test_public_dataset_adapter.py` covers the zero-based AI-TOD-v2 category
mapping, official invalid-box filter, crowd routing, and prediction inverse
mapping. `tests/test_aitod_official_evaluator.py` and
`tests/test_tinyperson_official_evaluator.py` lock both official protocols,
including TinyPerson uncertain-region IOD behavior and evaluator source hash.

## Legacy Evaluator Disposition

`common.eval_utils.evaluate` consumes every tile prediction/target pair as a
separate COCO image. `scripts/eval_checkpoint_original_image_folds.py` groups
tiles by original-image parity but still evaluates those tiles independently.
Neither is an original-image evaluator, and all metrics produced by those paths
remain diagnostic.
