# Public Benchmark Source Inventory

Status: `SOURCES_IDENTIFIED; AI-TOD-V2_ANNOTATIONS_ACQUIRED; IMAGES_AND_TINYPERSON_PENDING`
Checked: `2026-08-02`

## D1: TinyPerson

- Official project repository: https://github.com/ucas-vg/PointTinyBenchmark
- Official paper page: https://openaccess.thecvf.com/content_WACV_2020/html/Yu_Scale_Match_for_Tiny_Person_Detection_WACV_2020_paper.html
- Evaluator source commit: `bf6b83aa9a149ae15087eed4e9a7283f5cc67603`
  on the official `TinyBenchmark` branch.
- The repository is the current redirect target of the historical
  `ucas-vg/TinyBenchmark` URL and includes the TinyPerson benchmark/tooling.
- Repository code license: MIT. Dataset-specific distribution and annotation
  terms still require verification from the downloaded package.
- The updated official evaluator uses explicit `ignore`/`uncertain` handling,
  IoU thresholds `0.25/0.50/0.75`, `maxDets=200`, and TinyPerson-specific
  area bins. These settings are not interchangeable with stock COCO AP.

Before use, record the exact archive, checksum, split files, ignored regions,
category mapping, evaluator variant, and download date. Do not infer dataset
license from the code repository license.

## D2: AI-TOD-v2

- Official dataset site/repository: https://github.com/Chasel-Tsui/AI-TOD-v2
- Official implementation: https://github.com/Chasel-Tsui/mmdet-aitod
- Official evaluator fork: https://github.com/jwwangchn/cocoapi-aitod
- Implementation commit: `e3e5671114905ebd1b5f3da1bca86d07901533df`.
- Evaluator commit: `44a230ae5197cb89bf9e5e62f313cac3ad30c7af`.
- The implementation repository states that full train/validation/test sets are
  released and points to the dataset download and `cocoapi-aitod` evaluator.
- Repository code license: Apache-2.0. Dataset image/annotation terms still
  require verification from the downloaded package and its upstream AI-TOD/xView
  components.

Before use, record the exact archive, checksum, official split, eight-category
mapping, evaluator fork/version, area ranges, `maxDets`, and test-label access
  policy. The repository README contains an older note about held-out test
  annotations, so the downloaded artifact rather than that sentence must be the
  authority for actual availability.
- The official fork uses IoU `0.50:0.05:0.95`, `maxDets=[1,100,1500]`, and
  area bins `[0,8)`, `[8,16)`, `[16,32)`, and `[32,+inf)` pixels by square-root
  area.

## Dual Reporting Contract

Paper A keeps standard original-image COCO AP as its cross-dataset primary
outcome because that is the frozen claim target. Each benchmark's official
metrics are also reported from its pinned evaluator. They must be labeled as
separate protocols; official TinyPerson AP50/AP25/AP75 and AI-TOD-v2 AP@1500
must never be placed in a column labeled stock COCO AP@100.

## Freeze Rule

Identification of a repository is not dataset acquisition. G2 stays `REVISE`
until local immutable dataset manifests and evaluator-contract checks exist.
