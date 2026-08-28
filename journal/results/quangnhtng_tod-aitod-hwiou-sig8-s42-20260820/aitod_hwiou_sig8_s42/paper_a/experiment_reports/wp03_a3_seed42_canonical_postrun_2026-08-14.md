# WP03 A3 Canonical ALW Seed-42 Post-Run Audit

**Status:** `ARTIFACT_READY; PAIRED_COMPARISON_PENDING`

## Scope and boundary

This records only the first of the two frozen A3 validation-only runs. It does
not make a matched-method claim, update a result ledger, access final-test
material, or authorize any workload beyond the already-approved A3 SA-ALW
companion.

## Remote run and downloaded evidence

| Field | Value |
|---|---|
| account / kernel | `pptlyn11/wp03-a3-alw-canonical-s42` |
| API status | `KernelWorkerStatus.COMPLETE` |
| method / seed | `alw_canonical` / `42` |
| requested hardware | `NvidiaTeslaT4`; downloaded preflight reports Tesla T4 x2 |
| budget | 8 epochs, batch 4 |
| test access | validation only |
| artifact root | `.runtime/kaggle/wp03/a3_seed42/pptlyn11/outputs/shard_1/` |

The downloaded package contains the kernel log, `best.pt` (SHA-256
`a4d7d90b419e58a1a03670628aa9bcb48275906d9ebdfba7c9ede92e0e69ae42`),
config, eight-row epoch metrics, predictions, validation ground truth, and
both evaluator result families.

## Formal checks

- `audit_v8_output.py --method alw_canonical --seed 42 --tag wp03_a3`:
  `PASS` for frozen trainer/split/schedule, full epoch budget, selector,
  validation-only boundary, checkpoint, and non-empty predictions.
- Independent local CUDA strict reload: `PASS`.
  - trainer hash match: `7c05831c...b5d03`
  - paper-primary AP: kernel `0.1581097851`, local `0.1581537884`, delta
    `0.0000440033`
  - primary official maximum delta: `0.0001936087`, within the frozen
    `0.0005` tolerance
- Best validation checkpoint: epoch 6, AP `0.1581097851`, AP50
  `0.4341009448`, AP75 `0.0703988414`.

Secondary low-count official buckets exceed the primary tolerance in this
independent reload (`AP75_small` max `0.0016246821`). They are disclosure-only
under the frozen primary-endpoint rule and do not waive or alter that rule.

## Next action

Push the immutable serial companion
`pptlyn11/wp03-a3-sa-alw-full-s42` with the same private replica inputs and
T4 request. It must receive the same downloaded-artifact and independent
reload audit before A4 matrix/bootstrap work can begin.
