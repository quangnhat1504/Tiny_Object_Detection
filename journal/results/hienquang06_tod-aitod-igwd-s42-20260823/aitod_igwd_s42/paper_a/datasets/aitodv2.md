# AI-TOD-v2 Dataset Card

Status: `ANNOTATIONS_ADAPTER_EVALUATOR_TESTED; TEST_MATERIAL_DISCLOSED; IMAGES_PENDING; G2 NOT READY`

## Official Sources

- Dataset annotations:
  https://drive.google.com/drive/folders/1Er14atDO1cBraBD4DSFODZV1x7NHO_PY
- Dataset/implementation repository:
  https://github.com/Chasel-Tsui/mmdet-aitod
- Evaluator: https://github.com/jwwangchn/cocoapi-aitod
- Pinned evaluator commit:
  `44a230ae5197cb89bf9e5e62f313cac3ad30c7af`.

AI-TOD-v2 reuses AI-TOD images and provides revised annotations. The annotation
package alone is not a complete dataset; images still need to be acquired from
the AI-TOD/xView upstream path and hashed.

## Acquired Annotation Splits

| Split | Images | Raw annotations | SHA-256 |
|---|---:|---:|---|
| train | 11,214 | 301,534 | `ed7b37a1187b496b96943fa46c15aab39656d59eaf501192d85178354b637b2e` |
| validation | 2,804 | 75,091 | `d0439ba687db66bb38584a3398a906f15c47e055cdd84bb9060496a91f30d7cf` |
| trainval | 14,018 | 376,625 | `5d18a108b1e440d236a47669fd0a2ecf2cb5c57f588684c11975a03e28ecd262` |
| test | 14,018 | 376,121 | `9817a75f9bc4a84015881f2ddbf39bcc29635654a0266b210fd382743c927d98` |

Filename audit confirms train and validation are disjoint, trainval is their
exact union, and trainval/test are disjoint.

The test annotation file was structurally parsed during this audit, including
counts and box-size summaries. No predictions, checkpoints, or performance
metrics were evaluated. Therefore the split is performance-locked but must not
be described as literally unseen. Schedule fitting remains train-only. Further
access follows `../data_access_policy.md`.

## Categories and Loader Contract

Official category IDs are `0..7`: airplane, bridge, storage-tank, ship,
swimming-pool, vehicle, person, and wind-mill. Torchvision reserves label zero
for background, so the training adapter must map category `0..7` to label
`1..8` and invert that mapping when writing evaluator JSON.

The official MMDetection loader drops annotations with no image intersection,
non-positive area, or width/height below one pixel, and routes `iscrowd` boxes
to ignore targets. The Paper A adapter must match this behavior exactly.

## Official Evaluation

- IoU thresholds: `0.50:0.05:0.95`.
- Maximum detections: `[1, 100, 1500]`.
- Square-root area bins: very tiny `[0,8)`, tiny `[8,16)`, small `[16,32)`,
  medium `[32,+inf)` pixels.
- Original 800x800 benchmark images are evaluation units; no Paper A tiling is
  currently justified for this dataset.

The pinned official evaluator passes a perfect-box fixture locally with AP,
AP50, AP75, and tiny AP equal to `1.0`. Its historical build compatibility
shims are isolated and documented in
`../evaluation/aitodv2_runtime_compat.md`; the evaluator logic itself remains
hash-locked. This fixture must pass again on Kaggle before a training package
is released.

Machine-readable audit:
`aitodv2_annotation_audit.json`. Annotation-only manifest:
`../splits/aitodv2_annotation_manifest.csv`.
