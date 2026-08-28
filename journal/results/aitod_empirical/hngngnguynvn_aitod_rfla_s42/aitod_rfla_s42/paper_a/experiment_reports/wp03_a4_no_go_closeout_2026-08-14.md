# WP03 A4 Paper A NO-GO Closeout — 2026-08-14

**Decision:** `NO-GO — CLOSE PAPER A PERFORMANCE WORK`

## Decision boundary

This closeout applies the preregistered A4 rule in
`wiki/syntheses/strategic-research-roadmap-2026-08-14.md` to the complete,
validation-only TinyPerson matrix. It does not use final-test material and does
not authorize a rescue sweep, WP04/WP05, or an external matrix.

## Accepted matrix

All 18 rows in the six-method by three-seed matrix are accepted artifacts under
their documented audit/reload contracts. The machine-readable rows are in
`paper_a/results/wp03_a4_six_method_matrix.csv`.

| Method | AP mean ± std | AP50 mean ± std | AP75 mean ± std |
|---|---:|---:|---:|
| standard | `0.155572 ± 0.002641` | `0.434155 ± 0.005332` | `0.070974 ± 0.001899` |
| RFLA | `0.158568 ± 0.001379` | `0.441799 ± 0.005728` | `0.074839 ± 0.002418` |
| NWD | `0.148003 ± 0.001819` | `0.411355 ± 0.003320` | `0.067946 ± 0.001587` |
| IGWD | `0.149465 ± 0.000780` | `0.423776 ± 0.001456` | `0.066578 ± 0.001564` |
| canonical ALW | `0.156406 ± 0.001532` | `0.430988 ± 0.003359` | `0.069519 ± 0.003087` |
| full SA-ALW | `0.155120 ± 0.002356` | `0.426332 ± 0.003854` | `0.071202 ± 0.004322` |

The validation ground truth contains 2,041 original-image records and 4,719
non-ignore annotations: 3,951 COCO-small, 739 COCO-medium, and 29 COCO-large.
The count artifact is `paper_a/results/wp03_a4_scale_counts.json`.

## Direct paired comparison

| Seed | ΔAP | ΔAP50 | ΔAP75 |
|---:|---:|---:|---:|
| 42 | `-0.005646` | `-0.012218` | `-0.003538` |
| 123 | `+0.000796` | `-0.002814` | `+0.005153` |
| 2024 | `+0.000992` | `+0.001063` | `+0.003435` |
| mean | `-0.001286` | `-0.004656` | `+0.001684` |

Two of three seeds favor SA-ALW on AP, but the seed-42 regression dominates the
mean. SA-ALW also remains below the strongest accepted baseline, RFLA, on all
three headline means.

## Paired original-image uncertainty

The 2,000-replicate paired bootstrap samples the same original-image
multiplicities for both methods and all three fixed seeds. It exactly replays
the saved COCO metrics before resampling. The complete provenance artifact is
`paper_a/results/wp03_a4_paired_bootstrap.json` with SHA-256
`48d138af82db016491c57d8cd5f55e321fb7b44539ac2716de8c61f82c9403bd`.

| Metric | SA-ALW − ALW mean | 95% percentile CI |
|---|---:|---:|
| AP | `-0.001286` | `[-0.002939, +0.001277]` |
| AP50 | `-0.004656` | `[-0.009807, +0.000785]` |
| AP75 | `+0.001684` | `[-0.001778, +0.004946]` |

## Gate result

A4 requires the SA-ALW-versus-ALW AP interval to support a positive effect.
The observed AP mean is negative and its interval crosses zero. This directly
triggers `NO-GO`; the positive AP75 mean is uncertain and cannot override the
primary criterion.

Consequences:

- performance claims C009 and C010 are disabled;
- external generalization, placement, and component claims C011–C013 are closed
  without additional training;
- WP04, WP05, WP06, and WP07 are closed; no rescue ablation or sensitivity
  sweep is permitted;
- all accepted artifacts and negative evidence remain preserved; and
- compute may move only to the roadmap's Program B recovery audit.

## Final-test disclosure

The Paper A final-test state is unchanged. TinyPerson A3 material remains
unacquired/unopened, and no A4 final-test performance access occurred. The
existing AI-TOD-v2 structural provenance event remains a non-performance A3
access only. Thus the Paper A final-test performance-access count remains zero.
