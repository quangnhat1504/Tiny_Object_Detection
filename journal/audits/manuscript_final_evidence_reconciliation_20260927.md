# Final manuscript evidence reconciliation

Date: 2026-09-27 (Asia/Saigon). Scope: current `journal/manuscript/main.tex`, its compiled `main.pdf`, the existing Seed-42 validation records, and the already accepted official-test reconciliation. This is the current editorial/evidence index for the manuscript; it does not replace frozen pre-run or post-run records.

**Disposition: `REVIEW_READY_WITH_LIMITATIONS`.** The evidence-bounded draft is ready for internal scientific review. It is not certified for a venue or approved for submission. The current tables preserve explicit holds and selector qualifications. No retraining or new detector family is necessary for the claims that remain in the draft.

## Exact deliverables and verification

| Artifact | SHA-256 | State |
|---|---|---|
| `journal/manuscript/main.tex` | `9E3A77549503629B0724C869A032FCA4C7FFD78AE86076DA3DDD79FF628D60EF` | Current source |
| `journal/manuscript/main.pdf` | `851F77B31D620A0C7E601FCC72CD570FAB13020C1F4FDD8EC374BD004ECEB698` | Current 21-page A4 render; 6,229,952 bytes |

Two direct `pdflatex -interaction=nonstopmode -halt-on-error` passes completed in `.runtime/manuscript_caption_revision_20260927/`. The final log has no overfull/underfull boxes, undefined references, LaTeX/PDF warnings, emergency stop, or fatal error. MiKTeX printed its routine update-status notice; no update was attempted. All 21 A4 pages were rendered and visually reviewed after the caption revision; pages 10–11 were checked again at 72 dpi after tightening the Table 3 prose. Captions and nearby text show no clipping, overlap, or unreadable table content.

The current caption revision changes no reported score, table cell, selector, or figure asset. An earlier same-day revision removed the four-column AI-TOD-v2 block from Table 3 so the table is a focused TinyPerson comparison. AI-TOD-v2 validation diagnostics, including accepted IGWD/SimD rows and all related holds, remain in dated audits and are not presented as a complete comparative result.

## Evidence matrix: all 12 tables

| Table | Evidence disposition | Remaining qualification / action |
|---|---|---|
| 1. TinyPerson detector families | Qualified validation comparison. Cascade AP25 is 60.97 at shown precision from corroborating records within the frozen tolerance. | Cascade tiny1 AP50 stays withheld: replay records differ by 0.001871601, above 0.0005, and the second record lacks complete checkpoint/evaluator source binding. |
| 2. Homotopy forms | Six Seed-42 configurations use their best validation checkpoints and full allowed-validation replay. | Single seed; descriptive comparison only, no significance claim. Sigmoid is highest among the tested forms; the rational form is not presented as the winner. |
| 3. Bounding-box losses | TinyPerson-only comparison. Factorial values use best selectors. Eight external rows use terminal metric replay plus schema checks on manifest-bound best-checkpoint prediction artifacts. IGWD, GCD, and Inner-IoU use predeclared best-AP50 selectors and pass strict reload, schema, and full 1,684-tile replay. | The AI-TOD-v2 block is removed; its diagnostic values remain in audits because no protocol-matched baseline/proposed pair has a complete accepted identity, schema, and replay chain. For the eight TinyPerson terminal rows, metric and schema selectors differ; this is disclosed in the manuscript. The three best-selector rows use the same selector for both checks. Mixed selectors support no cross-method ranking. |
| 4. Faster R-CNN assignment | Standard IoU/HLA use accepted placement-factorial best selectors. ATSS/SimD use terminal replay; RFLA uses predeclared best AP50 epoch 6 and passes replay. Best-prediction schema artifacts pass the recorded checks. | ATSS/SimD schema artifacts are best-checkpoint outputs, separate from the displayed terminal selectors; this is stated in the prose before Table 4. NWD-assignment is withheld after terminal and best replay mismatch. All AI-TOD-v2 assignment cells are withheld for incomplete source/config and selector-bound replay identity. |
| 5. YOLO diagnostics | Native Ultralytics metrics from each run's last logged epoch, under a common native evaluator. | Not independent best-checkpoint benchmark replay; the manuscript labels the distinction and early stopping. Descriptive within-family results only. |
| 6–9. Official-test tables | Retain only the scope of the existing hash-bound reconciliation: 71 model rows / 575 cells. | No test inputs were reopened or re-evaluated here. Keep every official final-test surface closed; any extension requires a new explicit instruction and its applicable frozen queue/gates. |
| 10. Inference efficiency | Protocol-bound measurements over 1,684 allowed TinyPerson validation tiles on RTX 5070 Ti; 50 warmups and 1,684 timed inputs per model. | Forward-pass latency excludes loading, resizing, and host-to-device transfer as stated in the prose before Table 10. |
| 11. Faster R-CNN placement factorial | Four-arm Seed-42 placement comparison with full allowed-validation replay. | Single seed. The table separates RPN assignment from RoI regression; no universal effect claim. |
| 12. FCOS factorial | Eight Seed-42 arms from the frozen 2×2×2 protocol, best-validation selectors, and accepted full-validation replay. | Single seed. Zero AP for the GIoU/Gaussian-centerness arm is an observed protocol result, not a general failure claim. |

The older readiness r2 audit and the formula matrix retain their dated snapshots. Their Table 3 row dispositions predate the best-selector and schema-recovery addenda and are superseded for the current paper by this reconciliation and the cited row audits below. They were not rewritten.

## Figure and mathematical-claim review

| Figure | Disposition |
|---|---|
| 1. Geometry | SNGD translation illustration only. The adjacent prose scopes the uniform theorem bound to its compact, strictly separated domain and does not imply the plot shows the bound. |
| 2. Architecture | Uses the user-selected `fig2_architecture_publication.svg`, aligned to per-arm SNGD, target-normalized W2, and capped YOLO TW2-Y choices. |
| 3. Official-test overview | Kept within the prior 71-row reconciliation; no test input or result was recomputed in this pass. Points are described as tested checkpoints, not a statistical ranking. |
| 4. Placement | Matches the accepted Seed-42 Faster R-CNN placement result. |
| 5. FCOS factorial | Shows only the eight accepted TinyPerson arms; the held AI-TOD-v2 panel is excluded. |
| 6. Qualitative examples | TinyPerson validation only. The adjacent prose attributes differences to the Gaussian-centerness/TOD-sampling bundle and discloses increased false positives; held AI-TOD-v2 examples remain excluded. |

The formula and proof review found the current distinction consistent with the audited implementation routes:

- SNGD is symmetric and uses dimension-normalized center displacement plus log width/height ratios.
- TW2 divides squared Gaussian W2 by target area and is asymmetric in candidate/target. YOLO's TW2-Y caps the exponent and is excluded from the theorem.
- The rational scale gate is target-scale dependent. Entropy normalization explicitly requires `C_f > 1`; `beta = 0` recovers scale-only gating for the same affinity. Detaching entropy stops gradients through the entropy-prior branch, not all updates to shared detector parameters.
- The theorem's positive lower bound applies only in exact real arithmetic on a nonempty compact subset of strictly disjoint boxes, with nonzero bounded center displacement and bounded positive width/height ratios. It excludes contact, the YOLO cap, finite-precision saturation, and detector convergence. The corrective-direction proposition applies to centroid variables and does not guarantee an arbitrary network-parameter update will improve the detector.
- Dataset and protocol text distinguishes the audited TinyPerson train/validation surface, AI-TOD-v2 allowed validation, Kaggle training, and RTX 5070 Ti local replay/latency. The manuscript states Seed 42 only and does not claim variance or statistical significance.

No remaining mathematical contradiction was found in the scoped source review. The empirical paper does not contain a controlled SNGD-versus-TW2 comparison, so it must not attribute one arm's result to a particular affinity beyond the formula/source mapping already audited. A paired ablation is optional future work only if a later claim needs that causal distinction.

## Holds and next actions

- Keep Cascade tiny1 AP50, TinyPerson NWD-assignment, and all unaccepted AI-TOD-v2 assignment/loss rows withheld as currently shown.
- Keep the AI-TOD-v2 beta 0 / 0.5 pair out of the manuscript; the pair remains held for a beta-effect claim.
- Do not dispatch training for this revision. Existing evidence recovery closed the needed Table 3 rows, and the current claims do not depend on the remaining holds.
- Select a target venue before venue-specific checks. The final venue checklist must cover template/page limit, anonymization, supplementary-material rules, figure resolution/font embedding, ethics/data statements, and reference formatting.
- Verify the IGWD journal citation against an official publisher record before final submission. This review could not access the direct IEEE/DOI record; secondary indexes ([ResearchGate](https://www.researchgate.net/publication/402759726_Improved_Gaussian_Wasserstein_Distance_A_Smooth_Adaptive_New_Metric_for_Remote_Sensing_Tiny_Object_Detection), [EurekaMag](https://eurekamag.com/research/104/967/104967379.php)) list the same title/authors/year but are not treated as the authority for final bibliographic metadata. DOI: `10.1109/TMM.2026.3675527`.

## Source records

- `journal/audits/manuscript_submission_readiness_audit_20260927_r2.md` — prior readiness and 12-table inventory; its older Table 3 disposition is superseded here.
- `journal/audits/validation_table_recovery_plan_20260927.md` and `journal/audits/tinyperson_fcos_best_selector_validation_recovery_20260927.md` — selector/replay recoveries and immutable holds.
- `journal/audits/tinyperson_fcos_prediction_schema_recovery_20260927_r2.json` — schema evidence and explicit best-schema / terminal-replay selector fields.
- `journal/audits/tinyperson_frcnn_assignment_terminal_adjudication_20260927.md` and `journal/audits/aitodv2_frcnn_assignment_source_config_adjudication_20260927.md` — assignment evidence and unresolved AI-TOD-v2 provenance.
- `journal/audits/manuscript_arm_formula_evidence_matrix_20260926.md` — formula-to-arm map; older table dispositions are superseded by later recovery records.
- `journal/audits/manuscript_figure2_publication_source_alignment_20260927.md` and `journal/audits/manuscript_factorial_figure_replacement_20260926.md` — selected architecture and factorial figure provenance.
- `journal/audits/official_test_manuscript_reconciliation_20260925.json` — sole carried-forward scope for official-test tables; not re-run here.

No Kaggle API call, cloud mutation, training, held-out inference, official-test input access, repository `data/` access, or sealed-project access occurred in this reconciliation.

## Amendment: remove the sparse AI-TOD-v2 block from Table 3

At the user's direction, Table 3 is now TinyPerson-only. The AI-TOD-v2 columns were removed because the accepted table evidence contained only IGWD and SimD, without a fully accepted protocol-matched baseline/proposed comparison. No TinyPerson score or selector changed; the AI-TOD-v2 records and their row-level holds remain in the dated audits. The prior revision hashes were TeX `E03209A423BB872179FCA9D1039991FE65B3898FD88B925887A156A3F8A55BB6` and PDF `970039CE948874044E40A820F55B6B72EF0C17B2E2C7A7C53B392F5C5A334D84`; the deliverable table above records the current hashes.

The new PDF again has 21 A4 pages and was built with two direct `pdflatex -interaction=nonstopmode -halt-on-error` passes. The final log has no LaTeX/PDF warning or layout-error markers. Pages 10–11 were rendered at 144 dpi; the narrowed Table 3 is legible without clipping or overlap, has no empty AI-TOD-v2 columns or stale dash explanation, and Table 4 remains intact.

## Amendment: concise table and figure captions

At the user's direction, all 12 table captions and six figure captions were changed to short, descriptive titles, with the repeated bold title markup removed. Selector and replay qualifications, metric interpretation, measurement protocol, theorem scope, plot mapping, and the qualitative figure legend/results were moved into the adjacent manuscript prose. No table values, selectors, figure assets, or evidence dispositions changed. The forced page break before Table 4 was removed so its preceding discussion does not leave an isolated line on a nearly empty page.

The current deliverables are the hashes listed above. Two direct `pdflatex` passes succeeded; the final PDF is 21 A4 pages, and all pages were rendered and visually reviewed. No LaTeX/PDF, reference, overfull, or underfull warning was found. `scripts/audit_agent_contract.py` returned `PASS`. No training, Kaggle call, official-test access, validation replay, repository `data/`, or sealed-project access occurred for this editorial pass.
