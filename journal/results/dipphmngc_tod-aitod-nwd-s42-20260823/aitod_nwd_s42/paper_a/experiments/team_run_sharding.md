# Paper A Team Run Sharding

Status: `PREREGISTERED; UNASSIGNED`

The machine-readable handoff is `team_run_shards.csv`. No row authorizes a
Kaggle push until the user fills `owner` and `kaggle_account` after receiving a
separate pre-run report.

## Unit of Assignment

The atomic unit is a matched `dataset + seed` shard, not an arbitrary single
method. All methods in an atomic shard use the same dataset snapshot, code,
schedule, augmentation, data order, checkpoint rule, evaluator, and requested
GPU type. This preserves paired comparisons while allowing independent shards
to run on different accounts.

The required four-method pilot is paired with a two-method component shard
under the same owner/account/hashes. This six-method decision is preregistered
because beta-only has limited assignment paths and no regression path. After
both pilot shards pass the rule in `pilot_decision_protocol.md`, the six core
shards can run independently: three TinyPerson seed shards and three AI-TOD-v2
seed shards.

## Fair Workload Rule

Run counts are placeholders for planning, not a workload metric. Before team
assignment, the local real-batch smoke records peak VRAM and step time, and the
pre-run report converts these to predicted GPU-hours. Shards are then assigned
with longest-predicted-job-first balancing under these constraints:

1. Never split an atomic matched shard across accounts.
2. Target each member's assigned GPU-hours within 15 percent of the team mean.
3. Keep timing/efficiency measurements on the same T4 type and harness.
4. Do not compensate for a slow or failed account by changing epochs, batch
   size, accumulation, augmentation, or checkpoint selection.
5. Reassign an untouched shard freely; resume a started shard only from its own
   hash-matched checkpoint and record the migration in both reports.

## Dependency Order

1. `PILOT-D1-S42` plus `PILOT-COMP-D1-S42` decide G3.
2. The six core dataset/seed shards open only after pilot GO.
3. Component, placement, and sensitivity shards remain validation-only and open
   only after their exact grids and stopping rules are frozen.
4. Efficiency and qualitative audits reuse accepted checkpoints.
5. Final-test evaluation is intentionally absent from this board. It receives a
   separate one-pass package only after the release gates in
   `../data_access_policy.md` pass.

## Reporting Contract

Each shard has its own `experiment_reports/<shard_id>/pre_run.md` before push
and `post_run.md` after downloaded artifacts and independent reload. Combining
multiple shards into one conversational update does not replace these reports.
