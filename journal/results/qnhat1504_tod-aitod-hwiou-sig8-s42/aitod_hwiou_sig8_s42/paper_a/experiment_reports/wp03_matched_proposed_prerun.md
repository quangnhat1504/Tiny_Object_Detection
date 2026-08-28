# Paper A Pre-Run Report: `WP03` Matched Proposed-Method Matrix

Status: `APPROVED` — executing now.

## Purpose

WP03 establishes the matched proposed-method matrix for Paper A: the two
SA-ALW variants (pure ALW canonical + full SA-ALW) across three seeds
(42/123/2024). Seed 42 results are already available from WP01 pilot;
WP03 extends to seeds 123 and 2024 for the matched three-seed matrix.

## Frozen Method Set

Both methods use the same harness as WP01/WP02 (seed-matched, same data
order, augmentation, detector/backbone, validation-COCO-AP checkpoint
selector, original-image evaluator, fixed IoU-NMS):

1. **alw_canonical** — pure ALW (anisotropic + log-ratio + reliability +
   Charbonnier), placement `la_loss`, metric box loss. Reuse WP01
   `alw_canonical` config directly (already audited in WP01 pilot).
2. **sa_alw_full** — full SA-ALW (ALW + scale-adaptive beta + scale-adaptive
   position weight), placement `la_loss`, metric box loss. Reuse WP01
   `sa_alw_full` config directly (selected by WP01 gate decision).

Both methods use the frozen train-derived schedule (P10/P90 bounds,
beta_min=8.0, beta_max=10.0, w_min=1.0, w_max=1.5, linear form).

## Fidelity Audit Summary

- **ALW canonical**: already audited in WP01 pilot (selector AP 0.15461 at
  seed 42). Formula source: ALW paper (anisotropic position + log-ratio
  shape + reliability gate + Charbonnier smoothing).
- **SA-ALW full**: already audited in WP01 pilot (selector AP 0.15635 at
  seed 42, selected by frozen rule). Formula source: SA-ALW paper (ALW +
  scale-adaptive beta schedule + scale-adaptive position weight schedule).
- **Schedule**: frozen train-derived schedule from WP01 (schedule bounds
  P10/P90, audit sha256 locked). Both methods share the same schedule.
- **Trainer**: reuse WP02 trainer (hash `7c05831c...`), which includes all
  WP01 methods plus RFLA/NWD baselines. No trainer extension needed.

## Execution Assignment

- Owner: Qoder-Leader (user-delegated).
- Kaggle account(s): same pool accounts as WP02 (user-approved fan-out):
  `ngquangnht`, `amongus1504`, `thyngluthy`, `hienquang06`. (qnhat1504
  quota exceeded in WP02.)
- Kernel slug(s): one self-contained kernel per (method, seed) pair
  (`wp03-<method>-s<seed>`), created at push time.
- GPU request: `--accelerator NvidiaTeslaT4`, one T4 per kernel.
- Estimated runtime: same as WP01/WP02 (~4-5 h per method on T4 for 8
  epochs, batch 4, 6215 train records). WP03 uses the same reduced 8-epoch
  budget for comparability.
- Total kernels: 2 methods × 2 seeds = 4 kernels (seed 42 already done in
  WP01).
- Current owner load / team mean: WP02 in progress (6/12 COMPLETE, 6
  RUNNING); WP03 is the next Kaggle work package.

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

`APPROVED` — user approved execution on 2026-08-06. Trainer hash (reuse
WP02): `7c05831cbc544b84926694ecdd85159a9ac85ee557a7dc6894bebcfaed2b5d03`.

## Kaggle Package Record

- Data dataset: reuse WP01 `tinyperson-wp01-a1` (already uploaded and
  replicated per account). No new data packaging needed.
- Code dataset: reuse WP02 `paper-a-code-wp02` (already uploaded and
  replicated per account). No new code packaging needed.
- Staging manifest with per-file hashes: reuse WP02 manifest.
- No A3 test material in either package; validation-only scope preserved.

## Next Actions

1. Generate 4 kernels (`wp03-alw-canonical-s123`, `wp03-alw-canonical-s2024`,
   `wp03-sa-alw-full-s123`, `wp03-sa-alw-full-s2024`) via the same kernel
   generator pattern as WP01/WP02.
2. Fan out across the four-account pool (user-approved).
3. Post-run audit per kernel (artifact audit + off-kernel reload).
4. File WP03 post-run report with the matched proposed-method table.
