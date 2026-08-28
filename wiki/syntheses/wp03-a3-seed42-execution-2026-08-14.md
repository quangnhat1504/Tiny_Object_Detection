---
title: "WP03 A3 Seed-42 Execution State — 2026-08-14"
type: synthesis
tags: [paper-a, sa-alw, kaggle, validation-only, a3]
sources: []
last_updated: 2026-08-14
---

## State at handoff

The owner-approved A3 repair is a serial, validation-only seed-42 pair on
`pptlyn11`, using private replicas `tinyperson-wp01-a1` and
`paper-a-code-wp02`. It keeps the WP02 trainer SHA-256
`7c05831c...b5d03`, the PL-001 split hashes, 8 epochs, batch 4, and the
frozen schedule. It does not access final test data.

### Canonical ALW

`pptlyn11/wp03-a3-alw-canonical-s42` is complete and its output is downloaded
to `.runtime/kaggle/wp03/a3_seed42/pptlyn11/outputs/shard_1`. The artifact
audit and independent CUDA checkpoint reload both pass. The selected
validation AP is `0.1581097851` at epoch 6; local selector replay differs by
`0.0000440033`, and the maximum frozen primary-official endpoint delta is
`0.0001936087` (within `5e-4`). This is one accepted artifact, not a paired
method comparison or a promotion.

### Full SA-ALW

`pptlyn11/wp03-a3-sa-alw-full-s42` completed and was downloaded to
`.runtime/kaggle/wp03/a3_seed42/pptlyn11/outputs/shard_2`. Package audit and
independent CUDA reload pass. The selected validation AP is `0.1524633547` at
epoch 6; local selector replay differs by `0.0000897313`, and the maximum
frozen primary-official endpoint delta is `0.0001572373` (within `5e-4`).

## Decision boundary

Both seed-42 artifacts are accepted validation evidence. A4 subsequently
completed the frozen six-method by three-seed matrix and paired original-image
bootstrap. Its primary AP interval does not support a positive SA-ALW effect,
so Paper A is closed `NO-GO`; see [[WP03 A4 Paper A NO-GO — 2026-08-14]].

## Connections

- [[WP03 A3 Seed-42 Pre-Run — 2026-08-14]] — historical pre-run and frozen package contract.
- [[WP03 A2 T4 Re-adjudication Amendment — 2026-08-14]] — authorization basis.
- [[Strategic Research Roadmap — 2026-08-14]] — A4 and Paper A decision gates.
