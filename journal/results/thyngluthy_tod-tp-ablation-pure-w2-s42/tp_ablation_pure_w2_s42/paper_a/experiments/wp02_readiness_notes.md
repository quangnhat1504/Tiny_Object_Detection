# WP02 Readiness Notes — Faithful Baseline Integrations (draft, pre-GO)

Status: `DRAFT` — prepared in parallel with the WP01 pilot; WP02 stays
`BLOCKED` until WP01 returns GO. This memo inventories what already exists
in the shared harness and what a fidelity audit must settle before the
matched matrix (standard; RFLA; NWD; verified predecessor IGWD, seeds
42/123/2024) can get a pre-run report.

## What exists today

- NWD: `common/metrics/nwd.py` — diagonal-Gaussian W2, `exp(-beta * w2 / C)`
  with defaults `C = NWD_C = 12.0` and `beta = METRIC_BETA = 8.0`;
  registered as `"nwd"` in `common/metrics/__init__.py`.
- RFLA: full assignment path in `common/model.py` (receptive-field Gaussian
  similarity, dynamic top-k per scale `micro/tiny/small/large = 6/5/4/3`,
  quality ratio 0.60, `RFLA_K = 3`, `RFLA_BETA = 0.9` in `common/config.py`).
- IGWD (verified predecessor): `common/metrics/igwd.py`, already the frozen
  WP01 predecessor method; reuse carries over directly.
- Placement vocabulary (`build_model`): `everywhere`, `la`, `loss`,
  `la_loss`, `la_loss_nms`, `la_loss_soft_nms`, `saalw_assigner`.

## Fidelity audit questions (must be answered from primary sources)

1. **NWD sharpness**: official NWD (Wang et al., CVPR 2022) is
   `exp(-W2 / C)` — no extra beta multiplier. Our registry default applies
   `beta = 8.0`, giving `exp(-8*W2/12)`, a much sharper similarity. A
   faithful WP02 NWD baseline needs `beta = 1.0` (and the paper's own C
   choice documented per dataset). The legacy CBL-track NWD runs used the
   beta=8 form and must NOT be cited as faithful NWD numbers.
2. **NWD placement**: the official paper applies NWD to RPN label
   assignment AND NMS for tiny objects. Candidate faithful placement is
   therefore `la_loss_nms` (or `la` if the loss-side replacement was not
   part of their recipe); decide from the paper's ablation table, not from
   our legacy convenience.
3. **RFLA hyperparameters**: verify `k=3`, `beta=0.9`, the scale-band
   dynamic-k table, and the quality-ratio gate against the RFLA paper's
   TinyPerson/VisDrone configs; RFLA is assignment-only (their regression
   stays standard), so the faithful build is RFLA assignment +
   Smooth-L1 regression, i.e. metric in `la` placement with
   `box_loss_type="smooth_l1"`.
4. **IGWD**: already audited as the verified predecessor (formula source
   confirmed IEEE TMM 2026 in the 2026-08-02 formula audit); the WP01
   `igwd` config (`la_loss`, metric box loss) is exactly the integration
   WP02 reuses.
5. **Baseline fidelity matrix**: any NWD/RFLA configuration whose official
   method changes architecture, augmentation, or training schedule beyond
   the metric itself must be logged in
   `paper_a/experiments/baseline_fidelity_matrix.md` with the same
   default/fair/excluded taxonomy used for SWL/MMPW/DILA.

## Next actions (after WP01 GO)

- Resolve items 1-3 against `raw/NWD.pdf` and `raw/RFLA.pdf`, record
  decisions + sha of the consulted sources.
- Extend the pilot trainer (or a WP02 trainer fork) with the two baseline
  method configs; freeze hashes in the WP02 pre-run report.
- File the WP02 pre-run report before any kernel push, per
  `paper_a/experiment_execution_policy.md`.
