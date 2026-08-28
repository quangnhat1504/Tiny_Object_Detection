# WP03 A3 Matched Seed-42 Fill — Post-Run Audit

**Status:** `VALIDATION_EVIDENCE_ACCEPTED; A4_IN_PROGRESS`

## Scope and boundary

This report closes only the owner-approved A3 validation-only seed-42 fill.
Neither shard mounted or accessed TinyPerson final-test material. The two runs
use the frozen WP02 trainer, PL-001 split, eight-epoch schedule, batch size 4,
validation-AP selector, and fixed detector/NMS settings. Their acceptance opens
A4 matrix and paired-bootstrap analysis; it does not authorize WP04/WP05,
external training, or a performance claim.

## Remote execution and artifact status

| Method | Kernel | API | Hardware | Best epoch | AP / AP50 / AP75 |
|---|---|---|---|---:|---|
| canonical ALW | `pptlyn11/wp03-a3-alw-canonical-s42` | COMPLETE | Tesla T4 x2 | 6 | `0.1581097851 / 0.4341009448 / 0.0703988414` |
| full SA-ALW | `pptlyn11/wp03-a3-sa-alw-full-s42` | COMPLETE | Tesla T4 x2 | 6 | `0.1524633547 / 0.4218828090 / 0.0668611869` |

Both downloaded packages contain the kernel log, mount preflight, eight metric
rows, config, results, validation ground truth, non-empty saved detections, and
`best.pt`. The SA-ALW invocation label maps to canonical method
`sa_alw_canonical` in the frozen trainer.

## Audit and reload contract

Both `audit_v8_output.py --tag wp03_a3` package audits pass. Independent local
CUDA checks strictly loaded each checkpoint with trainer SHA-256
`7c05831cbc544b84926694ecdd85159a9ac85ee557a7dc6894bebcfaed2b5d03`.

| Method | Checkpoint SHA-256 | Local-vs-kernel AP delta | Max primary-official delta | Result |
|---|---|---:|---:|---|
| canonical ALW | `a4d7d90b419e58a1a03670628aa9bcb48275906d9ebdfba7c9ede92e0e69ae42` | `0.0000440033` | `0.0001936087` | PASS |
| full SA-ALW | `aa0866ee69bd179448721fcc7cf9c39c7a602bd54951d8128e6302f99df47404` | `0.0000897313` | `0.0001572373` | PASS |

The unchanged primary tolerance is `5e-4`. Secondary low-count official buckets
show larger disclosed environment drift (maximum `0.0016246821` for ALW and
`0.0019867826` for SA-ALW); these are outside the frozen primary-endpoint gate
and do not alter it.

## Ledger action and immediate observation

The two A3 runs now have matching entries in `paper_a/runs/manifest.jsonl` and
`paper_a/results/main_results.csv`. Together with the four PL-003-authorized v8
rows, the canonical ALW and full SA-ALW groups are complete at seeds
`42/123/2024` under the WP02 trainer.

At seed 42, full SA-ALW is below canonical ALW by `-0.0056464304` AP,
`-0.0122181358` AP50, and `-0.0035376544` AP75. This is one paired seed and is
not, by itself, the A4 decision.

## Next action

Run A4 exactly as frozen: regenerate the six-method by three-seed table, compute
paired original-image bootstrap intervals for SA-ALW minus ALW over AP/AP50/AP75,
and apply the roadmap GO/NO-GO rule. Keep final-test access and all additional
training closed while A4 is being adjudicated.
