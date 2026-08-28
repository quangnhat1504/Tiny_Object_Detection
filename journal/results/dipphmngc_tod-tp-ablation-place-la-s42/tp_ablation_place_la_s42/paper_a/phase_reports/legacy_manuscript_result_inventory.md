# Legacy Manuscript Result Inventory

Status: `QUARANTINED_DIAGNOSTIC`

The following files are historical drafts and are not part of the submission
workspace:

- `paper/saalw_main.tex`
- `paper/saalw_experiments.tex`

## Result-Bearing Locations

| Legacy location | Problem | Required disposition |
|---|---|---|
| `saalw_main.tex:49` abstract | Reused tile-level AP/AP50 and unsupported relative gain | Do not port; regenerate after G4 |
| `saalw_main.tex:79` introduction | Dataset scale statistics lack repaired provenance | Recompute from frozen train/source manifest |
| `saalw_main.tex:99` related-work gap | Universal prior-art claim is unaudited | Replace only after primary-source novelty matrix |
| `saalw_main.tex:111` ALW motivation | Uses first-metric and RMS wording | Correct in G1 method rewrite |
| `saalw_main.tex:134` contributions | Four bullets, superlatives, numeric test claims | Rebuild as at most three evidence-backed bullets |
| `saalw_main.tex:184` property table | Ambiguous properties and false SA-ALW invariance implication | Replace after G1 |
| `saalw_main.tex:299` mechanism observations | Infers mechanism from training curves without direct diagnostic | Replace with assignment/gradient artifacts |
| `saalw_main.tex:328` schedules | Missing explicit clipping in equations | Correct in G1 |
| `saalw_main.tex:359` limiting behavior | Claims recovery outside bounds while formula is unclipped | Correct and test in G1 |
| `saalw_main.tex:364` placement | Assignment behavior and beta effect are asserted without call-graph audit | Replace after placement audit |
| `saalw_main.tex:391` limitations | Treats public/multi-seed evidence as future polish | Keep as conference-readiness blockers |
| `saalw_main.tex:409` conclusion | Reused tile-level results and forbidden claims | Do not port; regenerate after G4 |
| `saalw_experiments.tex:7` dataset | Counts processed variants as independent images | Replace after G2 dataset audit |
| `saalw_experiments.tex:58` evaluator | Evaluates tiles and selects checkpoint by AP50 | Replace with original-image AP selector contract |
| `saalw_experiments.tex:66` headline table | Manually typed reused-test values; source values drift | Remove from submission path |
| `saalw_experiments.tex:135` component ablation | Wrapped ALW denominator and reused test | Rerun A0-A6 after G1/G2 |
| `saalw_experiments.tex:197` placement | Reused-test NMS/assigner claims | Rerun validation placement matrix |
| `saalw_experiments.tex:211` findings | Restates unsupported single-seed claims | Regenerate from ledgers after G4 |

## Numeric Drift

The legacy LaTeX and `runs/test_results.json` disagree on reported values for
nominally identical SA-ALW rows. Neither is authoritative. New tables must be
generated from `paper_a/results/*.csv`, whose rows must trace to
`paper_a/runs/manifest.jsonl`.

## Submission Rule

No text, number, table, or caption from the two legacy files is inherited by
default. Method prose may be reused only after G1 verifies it. Dataset prose may
be reused only after G2 verifies it. Result prose may be written only after G4.

