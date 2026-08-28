# WP03 Canonicality and Matching Audit

**Date:** 2026-08-14  
**Gate:** A1 `PASS` for implementation canonicality and seed-123/2024 matching  
**Evidence status:** `NO_PROMOTION`; A2 owner decision is still required

## Scope and authority

This is a read-only audit of the four downloaded WP03-v8 artifacts. The
authoritative chain is the generated kernel, its immutable `config.json`, and
the frozen trainer source; prose and legacy run labels are not method
authority. It compares `paper_a/method_spec.md`, `paper_a/placement_audit.md`,
the v8 artifacts, and the v12 checkpoint-replay manifests.

## Findings

| Question | Evidence | Finding |
|---|---|---|
| ALW denominator | The four artifacts record trainer SHA-256 `7c05831c...b5d03`, which matches `paper_a/tools/train_tinyperson_pilot.py`. Its `alw_canonical` branch calls `configure_metric("alw_canonical", beta=8.0)`. The canonical registry selects `sa_alw_canonical.compute_alw_similarity`, not a legacy wrapper. | **Pass.** The executed ALW is the unwrapped canonical squared-log-ratio formulation. |
| SA-ALW delta | The kernel argument `sa_alw_full` is a historical CLI alias. The frozen trainer maps it to `sa_alw_canonical`, then uses the locked linear P10/P90 schedule (`beta=8..10`, position weight=`1..1.5`). | **Pass, with naming correction.** The artifact is full canonical SA-ALW, not legacy `common.metrics.sa_alw`. |
| Placement and beta | The same trainer constructs both models with `placement="la_loss"` and `box_loss_type="metric"`. Canonical beta is used by assignment similarity; aligned regression distance accepts no beta. Position weight is the SA regression/assignment difference; NMS remains fixed IoU-NMS. | **Pass.** This matches the Paper A contract. |
| Matching controls | All four runs use TinyPerson PL-001 hashes `5bea...f026b` / `31d6...ab27`, 6,215/2,041 records, horizontal-flip-only augmentation, ResNet-50-FPN, 8 epochs, batch 4, EMA off, the same schedule hashes, and the same validation-COCO-AP selector. Only method and seed differ. | **Pass** for the 123/2024 paired comparison. |
| Artifact integrity | Every run has `best.pt`, detections, epoch metrics, results, and `validation_only` scope. The two ALW v12 input manifests hash the exact v8 checkpoint/config/prediction payloads; all four v12 replicas replay saved detections exactly and regenerate primary endpoints within `5.0623e-7`. | Integrity and T4 diagnostic replay pass; this is not an automatic promotion. |

## Required reporting corrections

The WP03 pre-run report's description of `alw_canonical` as using
"reliability + Charbonnier" is false for the executed artifact. It is
superseded for formula interpretation by this audit; the historical document
is retained unchanged. Likewise, `sa_alw_full` must be described as a legacy
*invocation label* mapped to canonical full SA-ALW, never as the legacy metric
implementation. `WP01` in immutable run IDs and shard metadata is lineage
metadata inherited by the pilot trainer, not evidence that a different method,
data split, or budget ran.

## Paired validation observations

| Seed | Canonical ALW AP | Canonical full SA-ALW AP | SA-ALW minus ALW |
|---:|---:|---:|---:|
| 123 | 0.15514145 | 0.15593770 | +0.00079625 |
| 2024 | 0.15596567 | 0.15695814 | +0.00099247 |

These values are validation-only reloaded paper-primary AP. They show a small
positive within-seed signal, not a completed three-seed result, a significance
claim, a baseline win, or novelty proof. Seed 42 remains unmatched because its
available WP01 pilot used a different trainer hash.

## Decision and next boundary

A1 passes because the method actually executed is traceably canonical and the
two seed pairs are controlled. It does **not** accept the original local ALW
reload failures or append a WP03 ledger row. The only next Paper A action is
A2: the project owner must explicitly approve or reject a superseding
reproducibility amendment that accepts the four immutable v12 T4 reports as
platform evidence for the two v8 ALW checkpoints while retaining the original
`5e-4` tolerance, original failure record, validation-only scope, and no
retroactive promotion. Without that approval, Paper A is `NO-GO` and the
roadmap pivots to CBL/PC artifact recovery.

## Reproduction checks

```powershell
.\.venv-cuda\Scripts\python.exe -m unittest paper_a.tests.test_alw_saalw paper_a.tests.test_train_scale_schedule paper_a.tests.test_saalw_anchor_assignment
Get-FileHash paper_a\tools\train_tinyperson_pilot.py -Algorithm SHA256
```

The test command passed 24 tests on 2026-08-14; the trainer hash was
`7c05831cbc544b84926694ecdd85159a9ac85ee557a7dc6894bebcfaed2b5d03`.
