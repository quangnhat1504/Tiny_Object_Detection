# Dataset Card: Current SOD-TinyPeopleInSea Derivative

Status: `DEVELOPMENT_ONLY; G2 NO-GO`

## Provenance Available Locally

- Roboflow project: `tinyperson`, version `5`.
- Export name: `RefinedTinyPerson-augmented-for-training`.
- Export date in local metadata: `2025-05-09 08:46 GMT`.
- Declared license: `CC BY 4.0`.
- Declared upstream work: TinyPerson / Scale Match for Tiny Person Detection.
- Local classes: `dry-person`, `wet-swimmer`.
- Local metadata states that masked and crowded areas were removed.
- Preprocessing: EXIF auto-orientation.
- Train augmentation in export: three variants per source using probabilistic
  horizontal flip, shear, and brightness adjustment.

Primary local sources:

- `data/README.dataset.txt`
- `data/README.roboflow.txt`
- `data/data.yaml`

## Count Semantics

The export contains `1,570` processed files, not `1,570` independent source
images:

- train: `1,374` processed files from `458` source IDs, exactly three variants
  per source;
- validation: `131` processed files from `131` source IDs;
- test: `65` processed files from `65` source IDs;
- total: approximately `654` source IDs.

These source-ID counts do not establish independent sequences. Video sequence
groups overlap heavily between the legacy splits.

## Known Protocol Failures

- `30` train-validation sequence groups overlap.
- `23` train-test sequence groups overlap.
- `20` validation-test sequence groups overlap.
- The old test has been accessed repeatedly and cannot become a fresh Paper A
  test through re-splitting.
- The current evaluator counts clipped tile GTs as independent images.
- Upstream original path, crowd, ignore, and mask provenance cannot be recovered
  from the export alone.

## Allowed Use

This derivative may be used for implementation debugging, mechanism checks, and
supplementary development context after group-disjoint repair. It cannot be the
main benchmark, the only generalization evidence, or a source of final test
claims.

## Required Public Benchmarks

- D1: official TinyPerson or official TinyPerson protocol.
- D2: AI-TOD-v2.

Their official licenses, download versions, annotations, ignored regions,
splits, and evaluator settings must be recorded in separate dataset cards before
G2 can pass.

