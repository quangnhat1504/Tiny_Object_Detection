# Paper A Test Access Log

## Legacy SOD Test

- Status: `REUSED; CLOSED FOR SUBMISSION CLAIMS`.
- Historical contents: 65 processed images / 826 evaluation tiles.
- Prior use: repeated checkpoint, metric, NMS, ensemble, and research-family
  evaluations.
- Disposition: diagnostic only; no further Paper A evaluation.

## Iterative-CBL Locked Test

- Status: `CONSUMED 1/1; CLOSED`.
- Scope: performance-research checkpoint, outside Paper A.
- Disposition: never used as SA-ALW submission evidence.

## Paper A Final Tests

- TinyPerson official final test: `NOT_ACQUIRED / NOT_OPENED`.
- AI-TOD-v2 test annotation material: `ACQUIRED / STRUCTURALLY_PARSED` on
  `2026-08-02` during annotation-package provenance work. The audit computed
  split counts, category counts, box-size summaries, and file hashes. It did
  not load predictions, checkpoints, or test metrics.
- AI-TOD-v2 final-test performance evaluation: `NOT_RUN`.
- Paper A final-test material-access events: `1` (AI-TOD-v2 structural audit).
- Paper A final-test performance-evaluation count: `0`.

The earlier shorthand `NOT_ACQUIRED / NOT_OPENED` for AI-TOD-v2 was inaccurate
and is retired. AI-TOD-v2 is performance-locked, not literally unseen. The
train-only schedule artifact reads only the train annotation file and records
its train-file hash; no test statistic may select a schedule, method, threshold,
checkpoint, fusion rule, or claim.

Before any Paper A final-test performance evaluation, record code commit,
dataset version, split hash, config hash, checkpoint selector, tiling, fusion,
evaluator, claim-ledger freeze, assigned owner/account, and the one-pass budget
in this file and `runs/manifest.jsonl`.

No ordinary pilot, core, ablation, sensitivity, or efficiency Kaggle package may
mount or copy final-test annotations. See `data_access_policy.md`.
