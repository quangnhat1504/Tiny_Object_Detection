# Paper A Pre-Run Report: `WP01` (TinyPerson G3 viability pilot)

Status: `READY_FOR_PUSH`

## Scientific Purpose

- Shard ID and source work package: `PILOT-D1-S42` + `PILOT-COMP-D1-S42`
  (both shards, one scientific decision), source `WP01`.
- Question: does any canonical SA-ALW variant show a positive validation
  signal over both the verified direct predecessor (IGWD) and pure canonical
  ALW, and does the two-schedule full method outperform its beta-only and
  position-only components?
- Gate and stopping rule: G3 per
  `../experiments/pilot_decision_protocol.md` (frozen selection rule,
  `0.001` AP tie band, simpler-variant preference order position-only >
  beta-only > full). Both shards finish before G3 is called.
- Paper claim affected: enables (or blocks) the matched three-seed core
  matrix; by itself it enables no performance claim.

## Frozen Inputs

- Dataset/version: TinyPerson official binary task-all,
  `erase_with_uncertain_dataset` train material (A1 only). No test archive
  and no test annotation is included in the Kaggle dataset package.
- Annotation/image/split hashes:
  - official corner annotation
    `tiny_set_train_sw640_sh512_all.json` sha256
    `8474f1242a072a4a6cfb55ef97c1b3661f2ab2ce2069670c8801a97aad5428bd`
    (8,256 crops, 32,430 positives; image archive hashes in
    `D:\paper_a_data\TinyPerson\acquisition_manifest.json`);
  - PL-001 split protocol frozen in `../protocol_ledger.md`;
  - `tinyperson_train_sub.json` sha256
    `5bea11d2d6c4f0e524455d7394492eff85991cb6140987573e8890806f9f026b`
    (6,215 crops / 27,711 positives);
  - `tinyperson_val.json` sha256
    `31d67f94a62d3d9ecbbf825a9dca0a21b22b1a297645dfc34402c59cab50ab27`
    (2,041 crops / 4,719 positives);
  - Kaggle dataset package sha256: recorded in the post-run report after the
    private upload (package contains only A1 material).
- Code commit/source hash: pilot trainer
  `paper_a/tools/train_tinyperson_pilot.py` sha256
  `38a8902323d91c48d3a0ea39fd92912940931422e55a519f2fbacb6c0634ec9f`;
  pinned TinyPerson official evaluator sha256
  `222b3173510e7a89bd03d077dce5d4a11e23ea6a7cd22afbbe930817b0886557`
  (`../evaluation/official_evaluator_lock.json`).
- Config hash: each run writes `config.json` with a `config_sha256` derived
  from the full frozen run config; the post-run report audits all six hashes
  for matched fields (everything identical except the method component).
- Methods and placements:
  1. `standard` — IoU assignment + Smooth-L1 regression
     (`metric_fn=None`, placement `everywhere`, box loss `smooth_l1`,
     box-loss warmup 0);
  2. `igwd` — verified direct predecessor (placement `la_loss`, metric box
     loss);
  3. `alw_canonical` — pure canonical ALW, beta `8.0` (placement `la_loss`);
  4. `sa_alw_canonical_beta_only` — beta schedule only;
  5. `sa_alw_canonical_pos_only` — position-weight schedule only;
  6. `sa_alw_canonical` — full SA-ALW.
  SA schedules use the train-only TinyPerson P10/P90 bounds
  (`s_min=7.4328`, `s_max=44.8468`, audit sha256
  `2ae4fb56...093ff9`), beta `8 -> 10`, position weight `1 -> 1.5`,
  linear form, per `../schedules/endpoint_protocol.md`.
- Seeds: `42` for all six runs.
- Epoch/update budget: reduced pilot budget `8` epochs, batch size `4`,
  SGD (lr `0.005`, momentum `0.9`, weight decay `1e-4`) with 2-epoch linear
  warmup from `1e-4` then cosine decay; identical for all six methods.
- Augmentation and transform: seeded random horizontal flip (p=0.5) as the
  only augmentation; torchvision `GeneralizedRCNNTransform` at
  min_size `640` / max_size `800` (matches the schedule-fitting coordinate
  system); no copy-paste, no weighted sampler; seeded per-epoch data order
  shared by all methods.
- Checkpoint selector: maximum validation `paper_primary_coco` AP
  (pycocotools standard COCO AP, IoU `0.50:0.05:0.95`,
  ignored/uncertain GT routed as `iscrowd=1`), evaluated per crop window;
  final metrics come from an independent strict checkpoint reload.
- Original-image evaluator and fusion: TinyPerson evaluation unit is the
  official crop window (no tile fusion applies). Two separately labeled
  families from the same predictions: `paper_primary_coco` (selector) and
  `benchmark_official` (pinned official ignore-aware evaluator, IoU
  `0.25/0.50/0.75`, maxDets `200`, tiny bins). Detector test-time NMS is
  the shared torchvision default (score `0.05`, IoU `0.5`, 200 dets/img).
- Test-access state: `validation_only`. TinyPerson A3 test material is not
  mounted anywhere in this package.
- Atomic matched methods: all six; `PILOT-COMP-D1-S42` uses the same
  account, GPU request, hashes, and budget as `PILOT-D1-S42`.
- Data-order pairing rule: identical seeded shuffle per epoch
  (generator seed `seed + epoch`), identical flip RNG seed, identical
  collate and loader settings for every method.
- Determinism disclosure: all RNGs seeded; cuDNN convolution kernels are not
  forced bit-deterministic, so single-GPU reruns could differ by kernel
  selection noise. No rerun is planned; the pilot is a within-harness
  matched comparison.

## Local smoke record

`paper_a/tools/train_tinyperson_pilot.py` passed a 1-epoch / 8-train /
4-val smoke on CUDA for all six methods (loss finite and decreasing,
checkpoint save + strict reload, both evaluator families emitted), plus a
perfect-detection fixture
(`.runtime/pilot_eval_path_fixture/report.json`: paper_primary AP = 1.0 and
official AP25/AP50/AP75 = 1.0 on a 64-crop val subset). Smoke checkpoints
were deleted; no pilot number is produced locally.

## Execution Assignment

- Owner: Qoder-Leader (user-delegated; pending explicit user confirmation
  of this report before push).
- Kaggle account: `ngquangnht` (first pool account; both shards on the same
  account per protocol).
- Kernel slug(s): one self-contained kernel per method
  (`wp01-pilot-<method>-s42`), created at push time.
- GPU request: `--accelerator NvidiaTeslaT4`, one T4 per kernel.
- Estimated runtime/GPU-hours: about 2-3 h per epoch per method on T4
  (6,215 crops, batch 4) plus ~15 min/epoch validation → roughly 20-25
  GPU-hours per method, ~130 GPU-hours total; each run fits one 12 h T4
  session only if the per-epoch wall time lands near the low estimate, so
  kernels save the best checkpoint and `metrics.csv` every epoch and are
  restartable from the last saved epoch.
- Current owner load / team mean: WP01 is the only active Kaggle package.

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
- Final-test material included: `NO`.

## Decision

`KERNELS_RUNNING` — all six pilot kernels are executing on Kaggle (see the
fan-out addendum below for the account map).

## Kaggle Package Record (addendum, 2026-08-04)

- Data dataset: `ngquangnht/tinyperson-wp01-a1` (private), version 2026-08-04
  12:04 UTC. Contents (zip-dir-mode): `erase_with_uncertain_dataset/` (746
  images, 167,810,210 raw bytes) + `splits/` (3 PL-001 JSONs, sha256 in
  `.runtime/kaggle/wp01/staging_manifest.json`).
- Code dataset: `ngquangnht/paper-a-code-wp01` (private), version 2026-08-04
  12:06 UTC. Contents: `common.zip`, `paper_a.zip` (trainer sha256
  `38a8902323d91c48d3a0ea39fd92912940931422e55a519f2fbacb6c0634ec9f`,
  matches local freeze), `pinned_evaluator.zip`, `torch_cache.zip`
  (fasterrcnn_resnet50_fpn weights sha256
  `258fb6c638b15964ddcdd1ae0748c5eef1be9e732750120cc857feed3faac384`).
- Staging manifest with full per-file hashes:
  `.runtime/kaggle/wp01/staging_manifest.json`.
- No A3 test material in either package; validation-only scope preserved.

## Multi-Account Fan-Out (amendment, 2026-08-04)

Kaggle enforces a 2-concurrent-GPU-session cap per account, so the user
authorized fanning the pilot across pool accounts instead of waiting in
waves. This amends the "both shards on the same account" assignment above;
all matched-condition guarantees are preserved because every run still uses
bit-identical dataset payloads (same staged files, same sha256 set), the
same frozen trainer/schedule/splits, seed 42, T4, and internet off; only
the Kaggle account (quota boundary) differs.

- Account map: `ngquangnht` = standard + igwd; `amongus1504` =
  alw_canonical; `qnhat1504` = sa_alw_beta_only; `thyngluthy` =
  sa_alw_pos_only; `hienquang06` = sa_alw_full.
- Each foreign account received private copies of both datasets with the
  same slugs (`<account>/tinyperson-wp01-a1`, `<account>/paper-a-code-wp01`),
  staged by NTFS junction from the audited staging dirs
  (`.runtime/kaggle/wp01/multi_account_push.py`).
- First fan-out kernel versions ERRORed because they launched before the
  per-account dataset versions finished processing (missing `torch_cache`
  at mount); re-pushed as kernel version 2 after `datasets status` reported
  `ready`, and all six report RUNNING.
- Cross-account status polling/output uses the token move-aside rotation
  helper `.runtime/kaggle/wp01/run_as.py` and the six-kernel poller
  `.runtime/kaggle/wp01/poll_all_kernels.py`.
