---
title: "WP03 A4 Paper A NO-GO — 2026-08-14"
type: synthesis
tags: [paper-a, sa-alw, decision, negative-result, bootstrap]
sources:
  - paper_a/experiment_reports/wp03_a4_no_go_closeout_2026-08-14.md
  - paper_a/results/wp03_a4_six_method_matrix.csv
  - paper_a/results/wp03_a4_paired_bootstrap.json
last_updated: 2026-08-14
---

# WP03 A4 Paper A NO-GO — 2026-08-14

## Decision

Paper A performance work is closed `NO-GO`. The complete canonical
ALW-versus-SA-ALW comparison uses seeds `42/123/2024`, the same WP02 trainer,
validation-only original-image evaluation, accepted package/reload evidence,
and a paired original-image bootstrap.

Full SA-ALW minus canonical ALW has mean AP `-0.0012859`; its 2,000-replicate
95% percentile interval is `[-0.0029388,+0.0012769]`. AP50 is also negative on
average (`-0.0046562`), while AP75 is positive on average (`+0.0016837`) but its
interval crosses zero. The preregistered primary AP criterion therefore fails.

## Consequences

- C009 and C010 are disabled; C011–C013 are closed without rescue training.
- WP04–WP07, an external Paper A matrix, and all Paper A final-test performance
  work are closed.
- Negative artifacts and canonical ledgers remain preserved.
- TinyPerson final-test material remains unacquired/unopened, and Paper A has
  zero final-test performance accesses.
- The only open roadmap action is Program B B0: read-only recovery of historical
  CBL/PC fair-20 artifacts, with no new kernel push or locked-test access.

## Artifacts

- `paper_a/experiment_reports/wp03_a4_no_go_closeout_2026-08-14.md`
- `paper_a/results/wp03_a4_six_method_matrix.csv`
- `paper_a/results/wp03_a4_scale_counts.json`
- `paper_a/results/wp03_a4_paired_bootstrap.json`
- `paper_a/results/bootstrap_ci.csv`

## Connections

- [[Strategic Research Roadmap — 2026-08-14]] — frozen decision rule and B0 path.
- [[WP03 A3 Seed-42 Execution State — 2026-08-14]] — final matched artifacts.
- [[Maximum-Performance Research Checkpoint - 2026-08-02]] — Program B history.
