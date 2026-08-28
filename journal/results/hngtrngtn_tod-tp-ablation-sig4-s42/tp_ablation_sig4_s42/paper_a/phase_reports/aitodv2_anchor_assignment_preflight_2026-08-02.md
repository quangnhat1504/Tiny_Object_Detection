# AI-TOD-v2 Anchor Assignment Preflight - 2026-08-02

Status: `PASS_TECHNICAL_PREFLIGHT; NOT_G1_SUBMISSION_EVIDENCE`

## Protocol

- Source: official AI-TOD-v2 train annotations only.
- Annotation SHA-256:
  `ed7b37a1187b496b96943fa46c15aab39656d59eaf501192d85178354b637b2e`.
- Selection: 64 images sampled from 11,203 eligible 800x800 train images with
  seed 42; labels were not used for image selection.
- Targets: 1,818 valid GT boxes transformed to 640x640 detector coordinates.
- Candidates: exact five-level Paper A anchor lattice, 306,900 anchors/image.
- Assigner: exact two-pass HLA used by canonical training.
- Schedule bounds: AI-TOD-v2 train P10/P90
  `6.1968/13.8564 px`; technical endpoints beta `8 -> 10` and position weight
  `1 -> 1.5` remain pilot candidates rather than performance-selected values.
- No image features, optimizer update, validation metric, or test material was
  used.

## Aggregate Results

| Variant | Positive anchors | Delta vs ALW | Covered GT |
|---|---:|---:|---:|
| ALW | 6,899 | - | 1,816 / 1,818 |
| Beta-only | 6,784 | -115 (-1.67%) | 1,816 / 1,818 |
| Position-only | 6,763 | -136 (-1.97%) | 1,816 / 1,818 |
| Full SA-ALW | 6,643 | -256 (-3.71%) | 1,816 / 1,818 |

Change decomposition against ALW:

| Variant | Changed anchors | Positive-set changes | Added | Dropped | Owner changes |
|---|---:|---:|---:|---:|---:|
| Beta-only | 127 | 125 | 5 | 120 | 2 |
| Position-only | 537 | 536 | 200 | 336 | 1 |
| Full SA-ALW | 593 | 590 | 167 | 423 | 3 |

Full SA-ALW changes mean positives/GT from `2.9103` to `2.8077` below
`s_min`, and from `3.8459` to `3.6899` inside the adaptive interval. Counts are
identical above `s_max`, confirming the clipped large-target recovery path in
this anchor audit.

## Interpretation

- Beta's empirical anchor effect is almost entirely positive-set eligibility,
  not ownership. This agrees with the monotonic-ranking proof.
- Position emphasis changes substantially more anchor decisions than beta.
- Full SA-ALW is more selective than either component, but it does not reduce
  GT coverage in this sample.
- Positive-count reduction alone cannot determine performance. It may improve
  localization quality or remove useful supervision; only the preregistered
  matched pilot can decide.

## Decision

Do not add a coverage-preserving variant before the G3 pilot. Retain full,
beta-only, and position-only as the three SA candidates. Repeat the same audit
on the acquired TinyPerson train split before launch; C007/C008 remain pending.

Machine-readable artifacts:

- `diagnostics/aitodv2_anchor_assignment_preflight.json`
- `diagnostics/aitodv2_anchor_assignment_by_scale.csv`
