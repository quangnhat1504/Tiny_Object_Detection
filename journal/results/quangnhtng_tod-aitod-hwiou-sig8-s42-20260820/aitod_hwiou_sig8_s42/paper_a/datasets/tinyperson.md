# TinyPerson Dataset Card

Status: `PACKAGE_ACQUIRED_AND_HASHED; ADAPTER_FIXTURE_ON_REAL_DATA_PASS; G2 NOT READY (AI-TOD IMAGES PENDING)`

## Frozen Task

Paper A uses the official binary `all` task: sea-person and earth-person are
merged into category ID `1`, named `person`. Raw two-category annotations are
rejected by the adapter. Dense images are excluded by the benchmark protocol.

Training must use the official `erase_with_uncertain_dataset` images and its
matching task-all annotations. The official benchmark erases ignore and
uncertain regions in training images; using raw images while merely dropping
their boxes would incorrectly train those regions as background.

## Adapter Contract

`tinyperson_original.py` accepts only the binary task-all category contract and,
for training, requires the official corner annotation. It crops the referenced
full erased image using `corner=[x1,y1,x2,y2]`; the shifted annotation boxes are
already expressed in crop coordinates. Normal boxes are positives. `ignore`,
`uncertain`, and `iscrowd` boxes are kept in the target's audit-only ignore
fields and never become training positives. Invalid or out-of-crop boxes are
dropped. Prediction label `1` is inverted to official category ID `1`.

## Official Evaluation

- Original full images are evaluation units.
- IoU thresholds are `0.25`, `0.50`, and `0.75`.
- `maxDets=200`.
- Area labels are `all`, `tiny`, `tiny1`, `tiny2`, `tiny3`, `small`, and
  `reasonable`.
- `uncertain` is treated as ignore, and detection-over-ignore matching uses
  intersection over detection area (IOD).

The hash-locked evaluator fixture passes AP25/AP50/AP75 at `1.0` while a
higher-scoring detection fully inside an uncertain region is ignored. Runtime
compatibility details are recorded in
`../evaluation/tinyperson_runtime_compat.md`.

## Acquisition Record (2026-08-04)

The official `tiny_set` package was acquired from the official Google Drive
link in the `ucas-vg/PointTinyBenchmark` TinyBenchmark README
(`https://drive.google.com/open?id=1KrH9uEC9q4RdKJz-k34Q6v5hRewU5HOw`),
downloaded through OneDrive split folders `tiny_set-20260804T095532Z-1-001/002`,
and restructured to the immutable local root:

```text
D:\paper_a_data\TinyPerson\tiny_set
```

Archive SHA-256 hashes, counts, license terms, and per-file annotation audit
are recorded in `D:\paper_a_data\TinyPerson\acquisition_manifest.json`.

Verified facts against the official dataset statistics table:

- erase train images: `717` labeled + `48` dense + `29` pure background = `794`;
- erase test images: `781` labeled + `30` dense + `5` pure background = `816`;
- `erase_with_uncertain_dataset/annotations/corner/task/tiny_set_train_sw640_sh512_all.json`
  is binary `{id:1, name:person}` with `8,256` corner entries, `32,430` normal
  positives, and `corner` on every image, matching the adapter contract;
- raw train/test annotations carry `ignore=3,369/1,989`-style and
  `uncertain=3,486` flags as documented.

Distribution terms recorded from the official dataset page: no social-internet
redistribution of clear-person images; identifiable people require
deidentification for demos/promotion; no commercial/business use. These terms
also restrict Paper A qualitative figures and any Kaggle hosting method, which
remains a pending user decision before pilot packaging.

Access classes: erase train material is A1; the test image archive and test
annotation files (`annotations/tiny_set_test*.json`,
`mini_annotations/tiny_set_test_all.json`, and erase test images) are A3
storage-only material and must not be mounted in pilot/core/sensitivity
packages. TinyPerson ships no official validation split; the validation policy
must be frozen through the protocol ledger before any method selection.

Real-data fixture verification (2026-08-04): the corner-task train annotation
loads through `TinyPersonOriginalDataset` with 8,256 records, 32,430 positives,
and zero corner-size mismatches; a 100-image decode sample (92 offset corners)
passed crop-size and box-in-crop checks; corner is confirmed `[x1,y1,x2,y2]`.
The pinned official evaluator smoke on a consistent 200-image train GT subset
with perfect detections returns AP25/AP50/AP75 `all` = `0.9889` and evaluator
SHA-256 `222b3173...` unchanged. Report:
`.runtime/tinyperson_real_data_fixture.json`. Full suite: `57 passed, 1 skipped`.

## Pending Work

Train-only scale bounds fitting in detector coordinates, the anchor/gradient
diagnostics, and the frozen validation-split policy remain pending. No model
run is authorized until G1/G2 configs freeze.
