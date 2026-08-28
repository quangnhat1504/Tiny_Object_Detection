# Paper A Pre-Run Report: `WP02` Matched Core Baselines

Status: `APPROVED` — trainer extended, awaiting kernel push.

## Purpose

WP02 establishes the matched core baselines for Paper A: the four methods
that form the scientific comparison set for the matched three-seed matrix
(seeds 42/123/2024). The pilot (WP01) gate returned `GO` with full SA-ALW
selected; WP02 locks the baselines against which SA-ALW will be compared.

## Frozen Method Set

All four methods use the same harness as WP01 (seed-matched, same data
order, augmentation, detector/backbone, validation-COCO-AP checkpoint
selector, original-image evaluator, fixed IoU-NMS):

1. **standard** — vanilla Faster R-CNN ResNet50-FPN, IoU assignment,
   Smooth-L1 regression. Reuse WP01 `standard` config directly.
2. **RFLA** — receptive-field Gaussian similarity assignment (top-k
   hierarchical), Smooth-L1 regression. Hyperparams from RFLA paper:
   `k=3`, `beta=0.9`, dynamic-k per scale `micro/tiny/small/large = 6/5/4/3`,
   `quality_ratio=0.60`. Placement: `la` (assignment only), regression
   stays Smooth-L1.
3. **NWD** — Normalized Wasserstein Distance (Wang et al., CVPR 2022).
   **Fidelity decision**: official formula is `exp(-W2/C)` with no extra
   beta multiplier. Our registry default applies `beta=8.0` (legacy CBL
   track); for faithful WP02 NWD, override to `beta=1.0`. Placement:
   `la_loss_nms` (RPN label assignment + NMS, per official paper recipe).
   Normalization constant `C=12.0` (dataset-specific, documented in NWD
   paper for tiny-object detection).
4. **IGWD** — verified direct predecessor (Hu, Chen, Tang, IEEE TMM 2026).
   Reuse WP01 `igwd` config directly (already audited as the frozen
   predecessor).

## Fidelity Audit Summary

- **NWD beta**: registry default `beta=8.0` sharpens official `exp(-W2/C)`
  into `exp(-8·W2/12)`. WP02 NWD must use `beta=1.0`. Legacy CBL-track NWD
  runs (beta=8) are NOT faithful and must not be cited.
- **NWD placement**: official paper applies NWD to RPN label assignment
  AND NMS for tiny objects. Faithful placement is `la_loss_nms`.
- **RFLA hyperparams**: `k=3`, `beta=0.9`, dynamic-k table, quality-ratio
  0.60 — all match the RFLA paper's TinyPerson/VisDrone configs. RFLA is
  assignment-only (regression stays Smooth-L1), so faithful build is
  `la` placement + `box_loss_type="smooth_l1"`.
- **IGWD**: already audited as verified predecessor (formula source
  confirmed IEEE TMM 2026 in 2026-08-02 formula audit); WP01 `igwd`
  config is exactly the integration WP02 reuses.
- **Baseline fidelity matrix**: any NWD/RFLA configuration whose official
  method changes architecture, augmentation, or training schedule beyond
  the metric itself must be logged in
  `paper_a/experiments/baseline_fidelity_matrix.md` with the same
  default/fair/excluded taxonomy used for SWL/MMPW/DILA.

## Execution Assignment

- Owner: Qoder-Leader (user-delegated).
- Kaggle account(s): same five-account pool as WP01 (user-approved
  fan-out): `ngquangnht`, `amongus1504`, `qnhat1504`, `thyngluthy`,
  `hienquang06`.
- Kernel slug(s): one self-contained kernel per (method, seed) pair
  (`wp02-<method>-s<seed>`), created at push time.
- GPU request: `--accelerator NvidiaTeslaT4`, one T4 per kernel.
- Estimated runtime: same as WP01 (~4-5 h per method on T4 for 8 epochs,
  batch 4, 6215 train records). WP02 uses the same reduced 8-epoch budget
  as WP01 for comparability; the matched matrix extends to three seeds but
  keeps the per-seed budget frozen.
- Total kernels: 4 methods × 3 seeds = 12 kernels.
- Current owner load / team mean: WP01 complete (AUDITED_GO); WP02 is the
  only active Kaggle work package.

## Expected Artifacts

- Protocol/config: `config.json` (with `config_sha256`) per run.
- Metrics and logs: `metrics.csv` (epoch, train loss, selector AP, official
  AP50, seconds, lr) and `results.json` (reloaded final evaluation, both
  families) per run.
- Checkpoints: `best.pt` per run (selector-epoch state dict) and
  `detections_best.json` (reloaded best-checkpoint detections).
- Failure artifact: kernel log and any partial `metrics.csv` if a run dies;
  reported, never silently restarted with changed config.
- Independent reload command: rebuild via
  `build_method_model(method, schedule)` +
  `load_state_dict(torch.load(best.pt)["model"], strict=True)`, exactly as
  the trainer's final phase does; the post-run report repeats this off-kernel.
- Final-test material included: `NO` (validation-only scope preserved).

## Decision

`APPROVED` — user approved fidelity decisions on 2026-08-05. Trainer extended
with RFLA and NWD method configs. Trainer hash:
`7c05831cbc544b84926694ecdd85159a9ac85ee557a7dc6894bebcfaed2b5d03`.

## Kaggle Package Record (pending)

- Data dataset: reuse WP01 `tinyperson-wp01-a1` (already uploaded and
  replicated per account). No new data packaging needed.
- Code dataset: extend WP01 `paper-a-code-wp01` with WP02 method configs
  (NWD beta=1.0 override, RFLA placement). New code dataset version
  `paper-a-code-wp02` (or version bump on `paper-a-code-wp01` if the
  trainer is extended in-place). Full hashes pending trainer freeze.
- Staging manifest with per-file hashes: pending.
- No A3 test material in either package; validation-only scope preserved.

## Next Actions (after user approval)

1. Extend the pilot trainer (or fork to `train_tinyperson_wp02.py`) with
   the two new baseline method configs (NWD beta=1.0 + `la_loss_nms`
   placement; RFLA `la` + `smooth_l1`).
2. Freeze trainer hash + method configs; record in this report.
3. Push code dataset (version bump or new dataset) with frozen trainer.
4. Generate 12 kernels (`wp02-<method>-s<seed>`) via the same kernel
   generator pattern as WP01.
5. Fan out across the five-account pool (user-approved).
6. Post-run audit per kernel (artifact audit + off-kernel reload).
7. File WP02 post-run report with the matched baseline table.
