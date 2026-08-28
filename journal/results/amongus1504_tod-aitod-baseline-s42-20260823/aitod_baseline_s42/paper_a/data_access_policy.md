# Paper A Data Access Policy

Status: `FROZEN 2026-08-02`

## Access Classes

| Class | Material | Allowed use before method freeze |
|---|---|---|
| A0 | Source pages, licenses, evaluator code | Provenance and implementation audit |
| A1 | Training images and annotations | Training, train-only schedule fitting, diagnostics |
| A2 | Validation images and annotations | Method/checkpoint selection and bounded sensitivity |
| A3 | Final-test images or annotations | Storage and checksum only after acquisition; no routine package access |
| A4 | Predictions evaluated against final test | One preregistered performance pass after all release gates |

Archive acquisition does not authorize extraction into a training workspace.
Kaggle pilot, core, ablation, and sensitivity datasets must contain A1/A2 only.
The final-test package is a separate work package, dataset mount, owner/account,
and post-run report.

## Current Disclosure

- TinyPerson A3 material has not been acquired or opened.
- AI-TOD-v2 test annotations were structurally parsed once during provenance
  audit. This is an A3 material-access event, but not an A4 performance access.
- The AI-TOD-v2 train schedule is derived only from
  `aitodv2_train.json` with SHA-256
  `ed7b37a1187b496b96943fa46c15aab39656d59eaf501192d85178354b637b2e`.
- AI-TOD-v2 must be described as performance-locked, not literally unseen.

## Final-Test Release Gates

All conditions are required before A4 access:

1. G1, G2, and G3 pass.
2. Code commit, environment, dataset/split hashes, and evaluator hashes freeze.
3. Method configurations and train-derived schedules freeze.
4. Validation-only checkpoint selector and inference/fusion rules freeze.
5. Claims ledger and one-pass test budget freeze.
6. A separate pre-run report names the owner, Kaggle account, expected
   artifacts, and independent reload command.
7. The user explicitly assigns that final-test work package.

After A4 access, no method, schedule, checkpoint, threshold, fusion rule, or
claim may be tuned and re-evaluated. A failed artifact copy does not authorize a
second metric run if the metric was already revealed; the event must be audited
from kernel logs and the access ledger.

## Package Rules

- Development notebooks fail closed when a final-test path or test dataset slug
  is supplied.
- Team members receive only the data class needed by their shard.
- Training shards use validation-only evaluation and report `test_access=none`.
- Efficiency runs reuse accepted checkpoints and use a frozen validation sample
  list; they do not create another test decision surface.
- Every post-run report states both material access and performance access.
