# Frozen Configuration Hold

## Current TinyPerson reference

The train-derived TinyPerson reference schedule is frozen for the approved
WP01-WP03 comparison only: P10/P90 bounds from
`paper_a/schedules/tinyperson_train_p10_p90.json` (audit SHA-256
`2ae4fb56295b9c3ad292ced030bd4472cfbc2a3eb63199e08cdfdf9099093ff9`),
`beta_min=8`, `beta_max=10`, `w_min=1`, `w_max=1.5`, and `linear` form.
The WP03 pre-run report pins the matching trainer SHA-256
`7c05831cbc544b84926694ecdd85159a9ac85ee557a7dc6894bebcfaed2b5d03`.

Each Kaggle run writes its immutable `config.json` and `config_sha256` into
the downloaded artifact; acceptance also requires the matching entry in
`paper_a/runs/manifest.jsonl`. This directory intentionally does not duplicate
those run-specific files.

## Limits

This is not a cross-dataset or final-test freeze. WP04/WP05 require separate
approved pre-run reports and their implementation/config hashes before any
Kaggle push. Canonical training must still fail when a required schedule value
is omitted.
