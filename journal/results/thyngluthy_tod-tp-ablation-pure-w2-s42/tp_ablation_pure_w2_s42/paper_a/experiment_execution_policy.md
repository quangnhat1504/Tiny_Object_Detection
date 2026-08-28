# Paper A Experiment Execution Policy

Status: `FROZEN 2026-08-02`

## Separation of Work

Paper engineering may proceed without an experiment launch. It includes:

- canonical method implementation and deterministic tests;
- dataset acquisition, hashing, provenance, and split audits;
- evaluator adapters and reference-fixture validation;
- manuscript structure, equations, related work, figure/table generators;
- run manifests, config generation, and Kaggle notebook preparation;
- bounded local smoke tests that establish execution correctness only.

Paper A training experiments run on Kaggle. No local training result is promoted
to a paper table, and no Paper A Kaggle experiment is pushed silently.

## Required Separate Reporting

Every Kaggle work package requires two user-facing reports separate from routine
paper-work updates:

1. Pre-run report before push: purpose, dataset/version/hash, methods, seeds,
   epochs, GPU request, code/config hashes, checkpoint selector, evaluator,
   account, owner, expected artifacts, and test-access status.
2. Post-run report after artifact download: kernel status, exact artifacts,
   independent reload result, failures, metric table, gate decision, and ledger
   updates.

A Kaggle URL or `COMPLETE` status is not a result. Metrics and failure artifacts
must be downloaded and independently checked.

## Local Smoke Boundary

Allowed local work:

- unit/property tests;
- synthetic forward/backward/AMP/reload checks;
- one or a few real batches to validate data/model/evaluator contracts;
- reference evaluator fixtures with synthetic predictions;
- no-update mechanism audits.

Local smoke output is labeled `technical_gate` or `validation_evidence`, never
`submission_evidence`. Multi-epoch performance comparisons are not local smoke.

## Assignment Rule

The user assigns each Kaggle work package to a team member and account. Codex
prepares self-contained notebooks/configs, performs local smokes, and audits the
returned artifacts. A row may move to `READY_FOR_PUSH` only when `owner`,
`kaggle_account`, dataset hash, code hash, config hash, and estimated compute are
filled.

Work is distributed using `experiments/team_run_shards.csv`. A matched
`dataset + seed` shard is atomic so account distribution cannot break the paired
comparison. Assignment is balanced by predicted GPU-hours from the same local
smoke, not by raw run count. The target maximum member load is the team mean
plus 15 percent unless a dependency or hardware constraint is documented.

No final test package is assigned until all method and validation decisions are
frozen. Test access remains one preregistered pass per official final protocol.
Material access and performance-evaluation access are recorded separately under
`data_access_policy.md`; ordinary training shards receive no final-test mount.
