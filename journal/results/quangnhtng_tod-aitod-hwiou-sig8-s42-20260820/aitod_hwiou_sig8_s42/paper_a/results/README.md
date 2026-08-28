# Paper A Result Ledgers

Status: `SCHEMA_FROZEN; WP02_VALIDATION_EVIDENCE_ACCEPTED=12`

These CSV files are the only numerical source for manuscript result tables.
They contain twelve accepted **validation-evidence** WP02 baseline rows; no
final-test or submission-evidence row exists. Legacy numbers must never be
inserted here.

## Files

- `main_results.csv`: matched core comparisons and external generalization.
- `component_ablation.csv`: A0-A6 component ablations.
- `placement_ablation.csv`: assignment, regression, and joint placement.
- `sensitivity.csv`: validation-only schedule sensitivity.
- `efficiency.csv`: accepted runtime, memory, and parameter audits.
- `bootstrap_ci.csv`: paired original-image bootstrap intervals.

Metric values use fractions in `[0,1]`, not percentages. A row becomes
`ACCEPTED` only after the associated Kaggle post-run report, artifact download,
independent checkpoint reload, evaluator audit, and matching entry in
`runs/manifest.jsonl` all pass.

Run `python paper_a/tools/validate_result_ledgers.py` before generating tables.
`build_result_tables.py` reads accepted rows only and never edits source
ledgers.

## Metric namespace

The accepted WP02 ledger rows use the reloaded `paper_primary_coco` metrics:
COCO AP/AP50/AP75 and area buckets on original validation images. The separate
`benchmark_official` TinyPerson endpoint is preserved in each artifact and may
be reported as a secondary benchmark result, but its AP50/AP75 values must not
be substituted into the primary ledger columns or mixed into their aggregate.
