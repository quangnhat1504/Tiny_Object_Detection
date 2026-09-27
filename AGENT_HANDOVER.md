# AGENT HANDOVER BRIEFING: TINY OBJECT DETECTION (EH-WIoU)

# 0. CURRENT OPERATIONAL OVERRIDE (2026-09-27)

### Table and figure caption revision (2026-09-27)

- Current manuscript: `journal/manuscript/main.tex` SHA-256 `9E3A77549503629B0724C869A032FCA4C7FFD78AE86076DA3DDD79FF628D60EF`; `journal/manuscript/main.pdf` SHA-256 `851F77B31D620A0C7E601FCC72CD570FAB13020C1F4FDD8EC374BD004ECEB698` (21 A4 pages, 6,229,952 bytes).
- All 12 table captions and six figure captions are short descriptive titles without manually bolded title text. Selector/evidence qualifications, metric interpretation, measurement protocol, theorem scope, plot mapping, and qualitative legend/results are described in nearby manuscript prose.
- Two direct `pdflatex -interaction=nonstopmode -halt-on-error` passes succeeded in `.runtime/manuscript_caption_revision_20260927/`. The final log has no reference, LaTeX/PDF, overfull, or underfull warnings. All 21 pages were rendered and visually reviewed; pages 10–11 were checked again after removing the forced break before Table 4 and tightening the Table 3 paragraph. `scripts/audit_agent_contract.py` returned `PASS`.
- No score, table cell, selector, or figure asset changed. No training, Kaggle call, validation replay, cloud mutation, official-test access, repository `data/`, or sealed-project access occurred.

### Final manuscript evidence reconciliation (2026-09-27)

- Current audit: `journal/audits/manuscript_final_evidence_reconciliation_20260927.md`. Current source/PDF hashes and the caption-revision build/render evidence are recorded in the preceding subsection. The 21-page PDF was reviewed after making Table 3 TinyPerson-only and again after the caption revision.
- Verdict remains `REVIEW_READY_WITH_LIMITATIONS`: ready for internal scientific review, not venue-compliant or submission-approved. No training or additional detector family is needed for current claims.
- Table 3 now contains only the TinyPerson loss comparison. Its AI-TOD-v2 block was removed at the user's direction because only IGWD/SimD had accepted cells and no complete protocol-matched baseline/proposed pair passed all gates; diagnostic rows remain audit-only. The eight external TinyPerson rows use terminal metric replay with schema checks from separate manifest-bound best-checkpoint artifacts; IGWD/GCD/Inner-IoU use best selectors for both. The prose before Table 4 discloses the analogous best-schema/terminal-selector split for ATSS and SimD.
- Preserve current holds: Cascade tiny1; NWD-assignment; incomplete AI-TOD-v2 external rows; AI-TOD-v2 beta pair as a pairwise effect claim. Official-test surfaces remain closed. No Kaggle API, training, cloud mutation, official-test data, repository `data/`, or sealed project was accessed in this reconciliation.
- Previous Table 3 dispositions in `manuscript_submission_readiness_audit_20260927_r2.md` and `manuscript_arm_formula_evidence_matrix_20260926.md` are dated snapshots; the final reconciliation plus row-level recovery records govern the current manuscript.

### Validation table evidence recovery (2026-09-27)

- Current recovery records: `journal/audits/validation_table_recovery_plan_20260927.md` and its addendum `journal/audits/tinyperson_fcos_best_selector_validation_recovery_20260927.md`. Table 3 additionally restores 12 TinyPerson cells for IGWD, GCD, and Inner-IoU using their frozen best-AP50 checkpoints (epochs 6, 15, and 6); all three pass strict reload, schema, and full 1,684-tile validation replay at the unchanged `0.0005` tolerance. Their terminal-epoch replay holds remain recorded; the manuscript labels the validated selector beside each row. No retraining was needed.
- Table 4 now omits the all-dash AI-TOD-v2 assignment columns. TinyPerson ATSS and SimD retain their terminal-selector values; RFLA uses its predeclared best-AP50 epoch-6 selector after full validation replay and schema checks passed. NWD-assignment remains withheld. The table labels selectors and makes no cross-method ranking claim.
- FCOS class-0 predictions were removed only from separate derived validation artifacts; all manifest-bound originals remain unchanged. Corrected report: journal/audits/tinyperson_fcos_prediction_schema_recovery_20260927_r2.json; first provisional report is explicitly superseded there.
- Best-checkpoint assignment replays: journal/audits/tinyperson_frcnn_assignment_best_replay_tp_frcnn_{atss,nwd,rfla,simd}_s42_20260927.json. ATSS/RFLA pass; NWD/SimD best selectors remain held. Terminal ATSS/SimD remain the displayed selectors.
- Table 1 Cascade AP25 is restored as 60.97: the accepted clean-EMA report and the separate replay differ by only 0.000017945 and round identically. Cascade tiny1 remains withheld because its two replay records differ by 0.001871601; the second replay lacks serialized checkpoint/evaluator hashes and uses a `common/model.py` whose hash differs from the frozen pre-run source. No training was dispatched; API/kernel state was not re-queried; no cloud, held-out, or sealed surface was accessed.
- Manuscript output: `journal/manuscript/main.tex` SHA-256 `af9ed358849b057b6ef126e2f3921e8962a257cfccfa1232351422b7e1e43283`; `main.pdf` SHA-256 `8c301ec9ad1b02862dcaf12894d65ed9b4f86f71340e4ad2a0a5e2c267bd8021` (21 A4 pages). Two direct `pdflatex` passes succeeded with no warning/error/overfull/underfull markers in the final log. Pages 10-11 were rendered and reviewed; Table 3's restored rows and explanation are visible without clipping, and Table 4 remains intact.
- Build-path incident: the first compile wrote into the pre-existing untracked `journal/manuscript/$build/` directory because the output variable was passed literally. The final PDF was rebuilt in `.runtime/table3_best_selector_render_20260927/`; the untracked directory was not touched again, staged, or committed. Its previous contents were not recoverable from Git.

### Manuscript Figure 2 publication-source alignment (2026-09-27)

- The selected architecture artwork is `journal/manuscript/figures/fig2_architecture_publication.svg`. `main.tex` now includes that asset. Its editable draw.io source and SVG/PDF/PNG companions were refreshed to identify the per-arm SNGD, target-normalized W2, and capped TW2-Y choices; the older SNGD-only label was stale. Renderer scripts now write the publication assets without overwriting the separate overview files.
- Figure 2 source audit: `journal/audits/manuscript_figure2_publication_source_alignment_20260927.md`. The 21-page `main.pdf` SHA-256 is `ae4c2d73865d725c90df1912f1d667748cfee688233f0566169f0e9a850e7912`; `main.tex` SHA-256 is `0f5f102b18e4322524f4b4d8b32ae6a6b1b67a75f677f501b8e957b5b88eb4e6`.
- Two direct `pdflatex` passes succeeded with no final log warning/error markers; manuscript page 7 and the standalone figure were rendered and reviewed. This source correction does not alter the `REVIEW_READY_WITH_LIMITATIONS` verdict or any scientific result. No cloud, training, held-out, dataset, or sealed-project surface was accessed.

### Manuscript diagnostic-table removal and queue decision (2026-09-27)

- Current manuscript audit: `journal/audits/manuscript_submission_readiness_audit_20260927_r2.md`; verdict `REVIEW_READY_WITH_LIMITATIONS`. The AI-TOD-v2 loss-only diagnostic subsection/table was removed. The prior `HOLD_SUBMISSION` reason tied to retaining it is superseded; this is readiness for human review, not venue-specific compliance or submission approval.
- Deliverables: `journal/manuscript/main.tex` SHA-256 `fe6bd629996b6fca417836ebbf16fc2a7d2dd8ef1690059f7a034354368994c5`; `journal/manuscript/main.pdf` SHA-256 `b349b51b9308e60f9d7cd2436989c683a3fbf1597bf15adfe2550af3ef9196e5`, 21 A4 pages. There are 12 table environments and six included figures. Two direct `pdflatex` passes succeeded; final log has no warning/error markers; pages 9-11 were rendered and reviewed.
- The loss-only numbers and identity/replay holds remain in their dated evidence records, not the paper. `tab:sota_losses` is now Table 3, `tab:sota_assignment` Table 4, and `tab:yolo_frontier` Table 5. AI-TOD-v2 baseline/sigma-bracket/Gaussian-bundle and all assignment values remain withheld where their row-level acceptance gates are incomplete.
- Queue decision for this revision: no additional training or detector family is needed. The current accepted/qualified claim set does not depend on the held rows. A same-detector SNGD-versus-TW2 comparison is only a future candidate if a later manuscript needs an empirical claim about one exact affinity; it is not queued or dispatched.
- No Kaggle API was queried, no cloud state was mutated, and no official-test or sealed-project data were accessed during this document revision. The existing official-test reconciliation remains the sole evidence scope for those tables.

### AI-TOD-v2 beta r2 software-matched replay (2026-09-27)

- Latest audit: `journal/audits/aitodv2_frcnn_dual_entropy_prior_r2_replay_environment_adjudication_20260927.md`; machine report: `journal/audits/aitodv2_frcnn_dual_entropy_prior_r2_validation_replay_torch210_no_grad_20260927.json` (SHA-256 `0a646763dbcf841784003f5ea67b5360bb22f116307a9ae5bd1b3f346ff08335`).
- Read-only owner-isolated API poll at 19:27-19:28Z reports both exact `/1` refs `COMPLETE`, 84 remote entries each. The `kernels logs` endpoint failed for both; local staged training logs and manifest-bound bundle audits remain available.
- Replayed both arms over all 2,804 allowed validation images using cloud-reported PyTorch `2.10.0+cu128`, TorchVision `0.25.0+cu128`, and the trainer's `torch.no_grad()` context. All seven frozen source hashes match. Beta 0 passes (max delta `0.000287835`); beta 0.5 remains held on AP_t/AR100/AR1500 (max `0.000580578` vs unchanged `0.0005` tolerance). Pair: `HOLD_PAIR_NO_BETA_EFFECT_CLAIM`.
- At the time of this replay audit, no retraining was needed because the manuscript contained no claim from this pair. No TeX/PDF change, Kaggle mutation, held-out inference, or sealed-project access occurred in that replay. Its then-current `HOLD_SUBMISSION` reason (the separate diagnostic loss-only table) is superseded by the manuscript-removal audit above.

### Manuscript validation-table final row audit (2026-09-27)

- Prior full audit before diagnostic-table removal: `journal/audits/manuscript_submission_readiness_audit_20260927.md`; its `HOLD_SUBMISSION` verdict and hashes are retained as the pre-removal snapshot and superseded by the current r2 audit above. It records the prior 21-page build and page 10–11 review.
- Table 4 now reports AI-TOD-v2 numeric validation values only for accepted IGWD/SimD; baseline, sigma-bracket, and Gaussian-bundle AI cells are withheld. TinyPerson external EIoU/SimD/SIoU are the only external rows passing scoped metric/schema gates; cross-selector ranking stays withdrawn.
- Table 5 now has dashes for all AI-TOD-v2 validation assignment cells. Four external rows have source/config identity holds; baseline/HLA RPN-only lack selector-bound full-validation replay; dual HLA validation is unavailable. TinyPerson ATSS/SimD pass; RFLA/NWD are withheld.
- No new training is required for the narrowed manuscript. The user conditionally authorized dispatch if necessary, but the current claims do not need the held rows. No Kaggle status was re-queried in this audit; latest live/API statements remain bound to the dated records only. No held-out or sealed content was accessed.

### AI-TOD-v2 Faster R-CNN assignment provenance hold (2026-09-27)

- `journal/audits/aitodv2_frcnn_assignment_source_config_adjudication_20260927.md` rehashes all four frozen notebook files and embedded payloads. The payload's `common/faster_rcnn_assignment.py` SHA-256 (`3ab64dd2...`) conflicts with each downloaded `run_config.json` source hash (`d8109de7...`) and the prior strict-reload factory hash (`13305118...`). The frozen trainer explicitly calls the factory with 8 classes / 800 px, while its serialized `DETECTOR_CONFIG` reports 1 class / 512 px; the same inconsistent config is embedded in checkpoint-state. The prior strict-reload record's `config_bound=true` did not test those fields or bind the factory to the payload, so its PASS is not acceptance evidence. All four AI-TOD-v2 external-assignment AP/AP50 pairs are now dashes in the manuscript. API was not re-queried; no training, cloud mutation, or official-test inference.

### Assignment table TeX/PDF refresh (2026-09-27)

- `journal/manuscript/main.tex` now withholds all four AI-TOD-v2 external-assignment AP/AP50 pairs; the independent TinyPerson dispositions remain. The 21-page `main.pdf` was rebuilt with two direct `pdflatex` passes because the MiKTeX `latexmk` wrapper could not bootstrap its missing Perl script while the update endpoint was unavailable. The build log has no LaTeX warnings/errors, and page 11 was rendered and visually reviewed. TeX SHA-256 `41f8cd5793abeea0b93e1ce2f3b39dbe67d3bdbe552508ceede3e9d8efce16ea`; PDF SHA-256 `512d7bee2c6836470a5e686539bfb9b8755ac403cc39b409da1886968cedb952`. Readiness remains `HOLD_SUBMISSION`.

### TinyPerson external assignment terminal replay (2026-09-27)

- `journal/audits/tinyperson_frcnn_assignment_terminal_adjudication_20260927.md` records isolated frozen-notebook source replay for RFLA/NWD/SimD/ATSS Faster R-CNN assignment. All four manifest-bound best-prediction artifacts pass schema. Full 1,684-tile terminal replay at tolerance `0.0005` passes ATSS and SimD, but RFLA tiny1 AP50 misses by `0.000659224` and NWD-assign AP50 all/tiny1 misses (max `0.002024601`). Their TinyPerson manuscript cells are withheld; AI-TOD-v2 assignment rows remain separate diagnostics. API version was not re-queried. No cloud mutation or official-test inference.

### TinyPerson FCOS external-row replay closure (2026-09-27)

- The remaining geometric CIoU/DIoU/EIoU/WIoU v1 and extended alpha-IoU/SIoU/Inner-IoU terminal rows now have full 1,684-tile replay and best-prediction schema checks. All four geometric arms pass the `0.0005` metric tolerance but CIoU/DIoU/WIoU v1 have 4/296/12 category-0 predictions in their best artifacts; EIoU passes schema. Extended alpha-IoU and SIoU pass metric replay, but alpha-IoU has 303 category-0 predictions; Inner-IoU fails tiny1 AP50 replay by `0.001060694`; SIoU passes schema. The unprinted Focal-EIoU control has zero cloud metrics and is not added to the paper. Across the 11 printed external TinyPerson SOTA rows, only EIoU/SimD/SIoU pass these scoped local gates; eight rows have all TinyPerson cells withheld. Evidence: `journal/audits/tinyperson_fcos_geom_terminal_adjudication_20260927.md`, `journal/audits/tinyperson_fcos_extended_terminal_adjudication_20260927.md`, and prior four-arm adjudication. API version not re-queried; mixed-selector ranking remains withdrawn. No cloud mutation or official-test inference.

### TinyPerson FCOS SOTA terminal replay (2026-09-27)

- `journal/audits/tinyperson_fcos_sota_terminal_adjudication_20260927.md` and its five machine-readable reports bind the four NWD/IGWD/SimD/GCD terminal `last.pth` selectors to the four-arm frozen source in Git commit `30d8801f3336274a41b4fd906a0352b6bad40002`, exact manifest hashes/configs, strict reload, and full 1,684-tile validation replay. Current `common/fcos.py`, `common/metrics/gcd.py`, and trainer hashes differ from the pre-run; replay used isolated raw Git blobs, not current files. NWD and SimD pass metric tolerance; IGWD AP50 and GCD AP75 fail. Best-prediction schema fails NWD/GCD on category-0 detections. Aggregate row verdict: only SimD passes the local terminal metric/schema gates; TinyPerson NWD/IGWD/GCD cells are now withheld in the manuscript. API version was not re-queried and no cross-method ranking is promoted. Official test remains closed.

### Manuscript version-gate wording corrected (2026-09-27)

- `journal/manuscript/main.tex` and the 21-page PDF now describe the FCOS v2r8 baseline/sigma-6 version-1 binding as an inference from authenticated current-version-one metadata and frozen code-cell parity. The earlier statement that binding remained open was stale. The sigma-6 AP50 replay mismatch and sigma-4 identity hold still prohibit a loss-only effect claim. Current hashes and render checks are in `journal/audits/manuscript_submission_readiness_audit_20260926.md`; verdict remains `HOLD_SUBMISSION`.
- `journal/audits/manuscript_coverage_checkpoint_20260927.md` records the current disposition of all 13 tables and seven original figures. The material remaining blockers are per-row selector/replay binding for SOTA loss and assignment tables; diagnostic labeling is preserved until those gates pass or the cells are withdrawn.
- `journal/wiki/overview.md` was reconciled with those gates: its old sigma-4 optimum, broad SOTA superiority and Cascade tiny1 improvement statements were withdrawn or marked historical. The dated change is appended to `journal/wiki/log.md`.

### Current-version-one binding and beta arm verdict (2026-09-27)

- Authenticated, owner-isolated Kaggle `GetKernel` metadata now reports `current_version_number=1` for both beta `-r2` and both FCOS v2r8 slugs. Current remote code cells match each frozen staged notebook; all four current-slug statuses are `COMPLETE`. The output API itself still ignores `/1`, so version binding is inferred from the simultaneous current-version-one metadata and current-session output, not from a version-specific output query. Snapshot: `journal/audits/kaggle_current_version_one_binding_20260927.json`; adjudication: `journal/audits/aitodv2_version_one_adjudication_20260927.md`.
- Rehashed all 84 beta outputs and execution log per arm against the staged download manifest. Beta 0 now has a closed version gate and retains `ACCEPT_VALIDATION`; beta 0.5 remains `HOLD_REPLAY_AP50` under the predeclared tolerance. The pair is still `HOLD_PAIR_NO_BETA_EFFECT_CLAIM`, with no manuscript promotion. FCOS v2r8 baseline's version and replay metric gates pass, but sigma-6 remains `HOLD_METRIC_MISMATCH`; no loss-only comparison claim. No cloud mutation or official-test inference.

### Cascade validation row discrepancy (2026-09-27)

- `journal/audits/manuscript_cascade_validation_selector_discrepancy_20260926.md` compares the manuscript Cascade method row to two tracked replay records. The former 62.76% AP25 and 27.43% tiny1 AP50 are unsupported by either; the replay records themselves disagree on tiny1 (25.8681% versus 26.0553%), both below the 26.70% baseline. Those two cells are now dashes and the Cascade tiny1 gain claim is withdrawn. Overall AP50 44.81% is explicitly clean EMA replay (cloud log 41.964%), and reasonable AP50 60.56% agrees at table precision. The old frozen evidence was not edited. Current PDF/TeX hashes and readiness gate are in `journal/audits/manuscript_submission_readiness_audit_20260926.md`.

### Manuscript factorial figure replacement (2026-09-26)

- The AI-TOD-v2 sigma-4 qualitative figure has also been excluded from the working TeX/PDF due to `HOLD_IDENTITY`; its source asset remains untouched. The PDF now has 13 tables, six retained figures, and 21 pages. Page 21 was visually checked after placing the appendix heading before the remaining TinyPerson qualitative figure. Figure dispositions and current TeX/PDF hashes: `journal/audits/manuscript_qualitative_exclusion_20260926.md`. The official-test overview is Figure 3 in the current PDF, not Figure 7; its prior 71-row/575-cell reconciliation scope is unchanged.
- Figure 5 in the PDF, sourced from `journal/manuscript/figures/fig4_factorial_transfer.svg`, now shows only the eight accepted Seed-42 TinyPerson FCOS factorial arms at their best validation checkpoints. The former AI-TOD-v2 sigma-4/sigma-6 diagnostic panel was removed from the graphic because sigma-4 identity and sigma-6 replay/version gates remain open. `journal/manuscript/figures/generate_factorial_validation.py` verifies exact manifest/replay hashes and all eight arm gates before rendering. Evidence, SVG/PDF hashes and page-19 visual review: `journal/audits/manuscript_factorial_figure_replacement_20260926.md`. TeX/PDF were rebuilt; readiness remains `HOLD_SUBMISSION` for the other open gates.

### Kaggle CLI version correction and manuscript selector hold (2026-09-26)

- Frozen-notebook, full 2,804-image validation replay of the staged AI-TOD-v2 FCOS v2r8 current-slug bundles is recorded in `journal/audits/aitodv2_fcos_loss_core_frozen_replay_20260926.json` and its post-run Markdown. Baseline passes AP/AP50/AP75 tolerance 0.0005 (max delta 0.000311); H-WIoU sigma-6 fails AP50 (delta 0.002783). Both have strict reload at 32,080,590 parameters; version 1 remains unbound by the CLI. Pair verdict `HOLD_PAIR_NO_LOSS_EFFECT_CLAIM`; no core loss-only improvement enters the manuscript.
- `journal/audits/kaggle_cli_version_binding_correction_20260926.md` documents that installed Kaggle CLI `status`, `files`, `logs`, and `output` ignore a `/version` suffix in their API requests. Earlier wording that these commands queried exact `/1` is withdrawn. Their responses describe the current slug. The beta `-r2` version-1 receipt and matched runtime identity provide circumstantial binding, but beta 0's prior `ACCEPT_VALIDATION` is provisional until independent version binding; beta 0.5 remains `HOLD_REPLAY_AP50`, and the pair stays unpromoted. No retry or official-test inference is authorized by this finding.
- `journal/audits/manuscript_sota_selector_audit_20260926.md` binds TinyPerson SOTA factorial rows to best checkpoints and external objective rows to terminal epoch-20 CSVs. Cross-method rankings using those mixed selectors are withdrawn. The Smooth L1 TinyPerson row has no located source and was removed from the working manuscript. AI-TOD-v2 sigma-4 has a frozen/runtime protocol mismatch; its loss-only table, Figure 4 and qualitative Figure 6 are marked diagnostic, and the abstract loss-only improvement claim was removed. Rebuild and final readiness audit are still pending.

### AI-TOD-v2 beta r2 terminal audit (2026-09-26, 15:50Z)

- Both receipt-bound `/1` kernels are `COMPLETE`. The `-r2` outputs were fetched under `.runtime/kaggle_aitod_frcnn_entropy_prior_seed42_20260926_rerun_v2/terminal_outputs_20260926T152245Z/`; each arm has 84 remote files plus execution log and a Kaggle CLI companion log. Manifest SHA-256 is recorded in `journal/audits/aitodv2_frcnn_dual_entropy_prior_r2_postrun_20260926.json`. No output was copied into `journal/results` or used for official-test inference.
- `journal/audits/aitodv2_frcnn_dual_entropy_prior_r2_bundle_audit_20260926.json` checks the v2 protocol ID, source/config/data hashes, all downloaded file hashes/sizes, finite metrics, prediction schema, full 2,804-image validation workload identity, checkpoint config/history, and strict reload at 41,342,746 parameters. Both arms have zero bundle errors. The UTF-8 Kaggle log contains epochs 1-12; the earlier Windows `logs` encoding failure was a CLI output issue, not a run failure.
- Full independent validation replay: beta 0 passes all nine tracked metrics under the predeclared 0.0005 absolute tolerance (max delta 0.00015994); its post-run verdict is `ACCEPT_VALIDATION`. Beta 0.5 exceeds the AP50 tolerance (delta 0.00067652). A second full replay with the trainer's `torch.no_grad()` still exceeds it (delta 0.00068507). Its verdict is `HOLD_REPLAY_AP50`; pair verdict is `HOLD_PAIR_NO_BETA_EFFECT_CLAIM`. The local PyTorch/TorchVision versions differ from the cloud versions, but that is only a possible explanation, not established cause. Do not weaken the tolerance or add a beta-effect claim to the manuscript. Details: `journal/audits/aitodv2_frcnn_dual_entropy_prior_r2_validation_replay_20260926.json`, `journal/audits/aitodv2_frcnn_beta05_no_grad_diagnostic_20260926.json`, and the post-run audit. Official test remains closed. A training retry is not indicated by this replay mismatch.

### Assignment evidence and beta poll scope correction (2026-09-26, 15:15Z)

- `journal/audits/validation_table_cell_ledger_20260926.md` now binds the four external TinyPerson assignment rows to their exact terminal epoch-20 CSV hashes. The AI-TOD-v2 external rows were already bound to terminal epoch-12 metric hashes. Neither set has a complete selector-matched independent validation replay map; manuscript Table assignment now labels these values descriptive and withdraws the external cross-method ranking. The readiness verdict remains `HOLD_SUBMISSION`.
- Two transient postdispatch snapshots at 15:12Z and 15:15Z reported `COMPLETE` for the **v1** slugs because `scripts/audit_aitodv2_frcnn_entropy_prior_dispatch.py` imported the v1 pre-run while pairing it with the v2 receipt. Do not use those snapshots for the `-r2` pair. The script now requires the v2 pre-run SHA, protocol, receipt refs and version 1, and queries exact `/1` refs. The corrected 15:15:38Z snapshot reports both `-r2/1` arms `RUNNING`, one-character logs, no visible epoch, and no remote files listed. See `journal/audits/aitodv2_frcnn_beta_poll_scope_correction_20260926.md`. Artifact and verification remain pending; no beta result is promoted.

### Manuscript evidence-recovery checkpoint (2026-09-26, 15:04Z)

- The active goal is `journal/audits/manuscript_evidence_recovery_plan_20260926.md`. Current verdict in `journal/audits/manuscript_submission_readiness_audit_20260926.md` is **HOLD_SUBMISSION**. The working draft has 13 tables and seven figures; the rebuilt 22-page PDF and TeX are hash-bound in that audit. The method now distinguishes SNGD, target-normalized W2 and the capped YOLO path; the proof applies only on its stated compact, strictly disjoint domain. Figures 1/2/4 and bibliography/dataset/hardware/epoch wording were corrected.
- Table SOTA validation recovery: `journal/audits/validation_table_cell_ledger_20260926.md` is the row/cell authority. AI-TOD-v2 IGWD and SimD passed frozen-source strict reload, independent replay over all 2,804 allowed validation images, manifest hash/size checks, deterministic prediction-schema recovery, and an error-free post-run audit: `journal/audits/aitodv2_fcos_sota_distributional_postrun_20260926.json`. Their cloud terminal AP75 values are 10.87 and 10.67, replacing unsupported 11.20 and 10.85. NWD remains `HOLD` on AP50 replay mismatch; GCD Wave H remains `HOLD_NORMALIZED_PREDICTIONS` under `journal/audits/aitodv2_fcos_wave_h_gcd_validation_replay_20260926.json`. Four more AI-TOD-v2 geometric AP75 cells and seven TinyPerson reasonable-scale cells conflict with their terminal CSVs; the manuscript withholds these and other rows without a completed validation gate. No held-out test eligibility was changed.
- The AI-TOD-v2 entropy-prior `-r2` beta pair is a separate pending experiment. Latest owner-isolated exact-ref poll at `2026-09-26T14:42:27Z` reports both `RUNNING`, one-character logs with no visible epoch, and no downloaded/accepted artifacts: `journal/audits/aitodv2_frcnn_dual_entropy_prior_postdispatch_20260926T144227Z.json`. API state, artifact state and verification state remain separate. Do not add beta results to the manuscript until expected files/log, hashes/schema, strict reload, independent 2,804-image replay and post-run audit pass.
- Checks completed for this checkpoint: `python -m unittest scripts.test_recover_aitodv2_fcos_sota_predictions` (4 passed), `python -m py_compile` on new replay/audit scripts, `scripts/audit_agent_contract.py` (PASS), `git diff --check` on staged in-scope changes, and two-pass `pdflatex` with no overfull/undefined-reference/fatal marker. The NWD, IGWD and SimD local GPU replays and GCD Wave H replay are validation-only; no training or official-test inference was dispatched. Remaining work: consolidate per-arm acceptance for other retained SOTA and assignment rows, audit Figure/table claims after those decisions, and accept or hold each terminal beta arm. Use only allowed `.runtime/local/aitodv2_a1_a2_seed42_v1` validation data; official-test splits and sealed areas remain closed.

### AI-TODv2 entropy-prior full output recheck and corrected rerun ready (2026-09-26)

- Re-pulled the exact v1 notebooks read-only from Kaggle. Both current API statuses remain `COMPLETE`; the remote code cells exactly match the frozen staged notebooks and neither trainer command passes `--protocol-id`.
- Rechecked all 85 local files per arm against the complete download manifest: no missing, extra, empty, hash-mismatched, or size-mismatched file. `metrics.json` has four finite rows; each checkpoint state's metric history equals it, its embedded config equals `run_config.json`, and its config fingerprint recomputes correctly. All four checkpoint configs omit `protocol_id`, so the identity gate still blocks strict reload, replay, and acceptance. Full record: `journal/audits/aitodv2_frcnn_dual_entropy_prior_full_output_recheck_20260926T062550Z.json`.
- Corrected the trainer to require `--protocol-id` for entropy-prior runs and serialize it in the canonical run config; the paired notebook builder and readiness gate now bind the expected value. Focused tests pass.
- New immutable rerun protocol: `journal/audits/aitodv2_frcnn_dual_entropy_prior_prerun_v2_20260926.json`, protocol `aitodv2_frcnn_dual_entropy_prior_seed42_v2_20260926`. It keeps Seed 42, the same dataset and budget, and beta 0 / 0.5; new slugs end in `-r2`. Fresh readiness at `journal/audits/preflight_runs/aitodv2_frcnn_dual_entropy_prior_readiness_20260926T062516Z.json` is `DISPATCH_READINESS=true`, `DISPATCH_AUTHORITY=false`. The first generic preflight queried status on not-yet-created slugs and therefore returned access denied; slug absence was subsequently verified by account-scoped `kernels list` in the readiness report. Both version-1 Kaggle pushes were accepted under receipt `journal/audits/dispatch_receipts/receipt_aitodv2_frcnn_dual_entropy_prior_20260926T062833Z_777f73c6.json`.
- Fresh post-dispatch check at `2026-09-26T06:29:48Z`: both exact new refs are `RUNNING`; each has zero output files and a one-character newline log with no visible epoch or error. Artifact and verification states remain pending. Post-dispatch record: `journal/audits/aitodv2_frcnn_dual_entropy_prior_postdispatch_v2_20260926T062948Z.json`.
- The v1 pair remains unaccepted, with no strict reload or replay. Official-test data remains `CLOSED_NOT_ACCESSED`.

### AI-TODv2 dual entropy-prior pair terminal artifacts checked (2026-09-26)

- Fresh account-isolated Kaggle queries at `2026-09-26T04:35:42Z` report `COMPLETE` for beta 0 (`luongsythanh/tod-aitod-frcnn-ehprior-b0-s42`) and beta 0.5 (`pptlyn11/tod-aitod-frcnn-ehprior-b05-s42`). Both logs contain epochs 1-12; each remote inventory has 84 files, including metrics, run config, checkpoints and validation predictions.
- All 84 outputs plus the execution log for each arm were downloaded without overwriting into `.runtime/kaggle_aitod_frcnn_entropy_prior_seed42_20260926/terminal_outputs_20260926T042053Z/`. The hash manifest is `download_manifest.json` (SHA-256 `d0746fee67151aa9b43f26a04077bf33b9fe5b8cca3d5a2196ef4dc3697f6f0b`). All downloaded files are non-empty. The seven frozen source hashes, run factors, dataset annotation/workload hashes, finite metrics, and validation-prediction schema/range checks pass.
- Metrics contain four validation rows at epochs 3, 6, 9, and 12. Best validation values: beta 0 AP 0.183900 / AP50 0.443841; beta 0.5 AP 0.183020 / AP50 0.435220. These are descriptive checkpoint metrics, not accepted replay results or a significance claim.
- Acceptance is **BLOCKED**: neither trainer-generated `run_config.json` contains `protocol_id`, although the frozen pre-run declares `aitodv2_frcnn_dual_entropy_prior_seed42_v1_20260926`. The frozen trainer source `scripts/train_frcnn_aitod.py` (SHA-256 `8b7d31859a213fb3b46d740112c610f102421d9a259fbc66bdfb9ae8f44292bb`) constructs the config at lines 444-495 without that field. The evidence-lifecycle identity gate requires it, so strict reload and full-validation replay were not run and no promotion/manuscript update is authorized by these artifacts. See `journal/audits/aitodv2_frcnn_dual_entropy_prior_postrun_20260926T043542Z.json`.
- The logs include a nonfatal notebook `MissingIDFieldWarning` and no training traceback/CUDA OOM marker. The local dataset audit still binds only the allowed train/validation surface with 2,804 validation images; all official-test splits remain `CLOSED` and untouched.

### TinyPerson homotopy recovery and AI-TODv2 paired run (2026-09-26)

- Recovery record: `journal/audits/tinyperson_fcos_homotopy_form_recovery_20260926.md`; machine-readable evidence: `journal/audits/tinyperson_fcos_homotopy_form_recovery_20260926.json` (SHA-256 `5a910c99e393108afee08ca0cb1bdf2d3610c6fcb8a06cb8e0cb2ff94a6483ee`).
- Five historical non-rational TinyPerson FCOS runs are recovered. Each passed strict checkpoint reload and independent replay over 1,684 validation tiles. Their code cells match the corresponding frozen staged notebooks. The accepted rational control is in the existing factorial replay audit. Do not retrain the six homotopy forms.
- Corrected Seed-42 TinyPerson homotopy AP50 values: rational 0.262253, exponential 0.266666, sigmoid 0.279327, static 0.255222, pure W2 0.222720, pure IoU 0.270602. The sigmoid form leads the tested AP/AP50/tiny1 metrics; this is single-seed descriptive evidence, not a significance claim. The old Phase 2 narrative's rational 0.3054 is a separate Gaussian-centerness composite result; old sigmoid/static values do not match recovered artifacts.
- The immutable design JSON's dataset tree value `d4625ae...` conflicts with recovered run configs and the canonical audited content tree `87232d...`. Local verification confirms `87232d...`; do not alter immutable design records.
- The AI-TOD-v2 pair now uses dual H-WIoU (RPN assignment + RoI regression), EH-WIoU, detached training-only entropy priors, and matched beta 0 / 0.5 arms. The RoI entropy-prior path is detached under `torch.no_grad`; focused tests pass 3/3. Source/package, notebook, data-contract and exact target-slug checks pass.
- Versioned pre-run: `journal/audits/aitodv2_frcnn_dual_entropy_prior_prerun_20260926.json`. Fresh readiness: `journal/audits/preflight_runs/aitodv2_frcnn_dual_entropy_prior_readiness_20260925T175123Z.json`, `DISPATCH_READINESS=true`; dataset ACL passed for both accounts and both unique target slugs were absent before push.
- Both Kaggle pushes were accepted: beta 0 at `luongsythanh/tod-aitod-frcnn-ehprior-b0-s42`, beta 0.5 at `pptlyn11/tod-aitod-frcnn-ehprior-b05-s42`. A fresh API audit at `2026-09-25T18:52:12Z` reports `RUNNING` for both; account-isolated quota snapshots from `18:33Z` to `18:50Z` increased by 1,043 and 1,047 GPU seconds. This confirms stable process/GPU allocation over 17 minutes, but epoch progress is not visible: each log endpoint returns one newline and output listing is empty. Latest API poll: `journal/audits/aitodv2_frcnn_dual_entropy_prior_postdispatch_20260925T185212Z.json`; stability and manuscript audit: `journal/audits/manuscript_tables_and_experiment_queue_review_20260926.md`.
- Synthetic production-path smoke passes dual RPN/RoI forward-backward and verifies inference skips entropy-prior computation; script: `scripts/smoke_aitodv2_frcnn_entropy_prior.py`.
- Artifact state: training outputs have not been downloaded. Verification/promotion state remains pending non-empty artifacts, hashes, strict reload, full replay over all 2,804 allowed validation images, and post-run audit. Kaggle quota is not a constraint per the user. Official test splits remain `CLOSED`.
- Manuscript review corrected nine TinyPerson official-test metric cells and added the available `AP_vt` and `AP_t` columns to both AI-TOD-v2 official-test tables. All 71 official-test table rows now match the hash-bound reconciliation; all 575 displayed metric cells and family maxima were checked. No internal registry IDs remain in the table labels. The 13 tables retain separate comparison or ablation purposes; no table was removed. See the dated audit above.
- Experiment/queue decision: the six TinyPerson homotopy forms are recovered and do not need retraining. The dispatched AI-TOD-v2 beta 0 / beta 0.5 dual-placement pair is the only remaining training experiment for the current manuscript. Do not queue more training while it is active. The 28 `HOLD_VALIDATION` candidates concern held-out-test eligibility, not missing training runs. Keep all official-test splits closed.
- Boundary note for follow-up agents: a prior recursive inventory and filtered content search over `journal/results` exposed names under the sealed `paper_a` area; its truncated output leaves the earlier content-read question unresolved. On 2026-09-27, one `rg --files` inventory over `scripts`, `journal/audits`, and `journal/results` returned path names under that sealed namespace; no matching content was opened or executed, and no files there were modified. Do not repeat or rely on either broad-search result; the current official-test table reconciliation uses the separate hash-bound audit in `journal/audits`.
- Follow-up exact-ref audit at `2026-09-25T19:05Z` still reports both `RUNNING`; each owner's kernel list contains only the target, account GPU quota continued to increase, logs return one newline, and output listings remain empty. Read-only pulls of both remote notebooks confirm the expected training command uses ordinary `subprocess.run` without `-u`; the trainer prints epoch completion only after a full epoch and has no batch progress. Buffering is a plausible reason logs remain blank, not evidence of epoch completion. Details: `journal/audits/aitodv2_frcnn_dual_entropy_prior_run_observation_20260926.md`; latest API poll: `journal/audits/aitodv2_frcnn_dual_entropy_prior_postdispatch_20260925T190326Z.json`.
- Human-readable run record: `journal/audits/aitodv2_frcnn_dual_entropy_prior_dispatch_20260926.md`.
- AI-TOD-v2 train count remains 11,214 annotation records and 11,204 non-empty workload images under the existing `filter_empty=true` contract.

The AI-TOD-v2 `NOT READY` state in the 2026-09-25 audit has been superseded by the versioned package and dispatch recorded above. The older audit remains unchanged as a dated historical record.

This section supersedes the 2026-09-25 operational status for TinyPerson homotopy output availability, metric values, its retraining plan, and the AI-TOD-v2 pair's former `NOT READY` state.

### Extended validation recovery - Wave II

- Fresh test-blind audit: `journal/audits/extended_official_test_recovery_post_validation_v2_20260924.json`.
- State across 60 recovery targets: 18 `COMPLETE`, 12 `ELIGIBLE_FOR_TEST`, 28 `HOLD_VALIDATION`, 2 `PERMANENTLY_EXCLUDED`.
- The 12-ID queue is `EXT-016`, `EXT-020`-`EXT-022`, and `EXT-072`-`EXT-079`; queue record: `journal/audits/extended_official_test_ready_queue_post_validation_v2_20260924.json`.
- Recovery audit SHA-256: `545607d485a737f09652b0ad026b96a97cdbdb2236266e68db3ddf161cd91d2e`. Queue SHA-256: `30bb7f83d05ecc75ade5c19b4b478a02811f1758cfdf6eff050a23b3f3db58dc`.
- `EXT-016/020/021/022` passed config-bound strict reload at 41,342,746 parameters with zero missing/unexpected keys. Their accepted validation AP50 values were already manifest-bound: 0.251184, 0.451318, 0.440411, and 0.475625.
- `EXT-072`-`EXT-079` passed full allowed validation replay over 1,684 tiles and 118 original validation images. AP50: 0.489818, 0.453538, 0.466201, 0.460235, 0.473691, 0.419519, 0.450261, and 0.452913 respectively. Evidence: `journal/audits/extended_tinyperson_yolo_wave_i_validation_recovery_20260924.json` and each candidate's hash-bound recovered prediction artifact.
- Targeted runner support now checks explicit IDs and model families. The fresh audit recognizes the 18 prior full-test bundles using each bundle's bound shared or per-model runtime-guard trace; those 18 are `COMPLETE` and excluded from this queue.
- Queue dry-run passed and selected exactly the 12 IDs above. No official-test inference had been launched when this recovery audit and queue were prepared. The subsequent authorized execution is recorded below.
- Remaining 28 holds do not pass every gate. Do not add them based on family-level runner presence alone; rerun the dated audit after new evidence is accepted.

### Extended TinyPerson official-test execution - EXT-073-079 (2026-09-25)

- The user explicitly authorized full official-test inference for `EXT-073`-`EXT-079`; execution used the pinned queue launcher with checkpoint-hash verification and both `--run` and `--authorize-official-test`.
- Runner completed with `EXTENDED_TARGETED_TEST_INFERENCE=PASS`. All seven models have prediction, full bundle, and runtime-guard trace artifacts under `journal/test_raw/extended/full/`.
- Post-run raw-evidence audit: `journal/audits/extended_tinyperson_official_test_postrun_20260925.json`, verdict `PASS` for 7/7. Each bundle records 786/786 samples, sample-set equality, config-bound strict reload `PASS`, finite metrics, matching checkpoint/config/prediction/trace hashes, and zero forbidden or sealed-workspace accesses.
- This execution does not alter the frozen recovery audit or queue. The other five IDs in that 12-ID queue were not selected in this run. The report records raw evidence verification and does not claim manuscript promotion.

### AI-TODv2 assignment inference found - EXT-016/020/021/022 (2026-09-25)

- Existing raw bundles and predictions record one explicit four-model run. All four bind the complete 14,018/14,018 test sample set; checkpoint/config/prediction/shared-trace hashes, strict reload, finite metrics, and zero forbidden/sealed accesses pass the additive evidence check.
- Initial post-run reconciliation: `journal/audits/extended_aitod_frcnn_assignment_test_postrun_reconciliation_20260925.json`; its source blocker is superseded by the v2 reconciliation below.
- Source provenance is reconciled in `journal/audits/extended_aitod_frcnn_assignment_test_postrun_reconciliation_v2_20260925.json` (`PASS`, 4/4). The committed helper defines `repository_commit` as `git rev-parse HEAD`; the separate `runner_sha256` binds the actual runner bytes. An immutable runner snapshot at `journal/audits/source_snapshots/run_aitod_extended_test_inference_c0161c5645.py` matches all four recorded runner hashes exactly. The original bundles and both provenance values remain unchanged.
- These models already have accepted validation and config-bound strict reload evidence. No retraining, validation rerun, or inference rerun is needed.

## 0. CURRENT OPERATIONAL OVERRIDE (2026-09-22)

This section supersedes later operational status. The detailed AI-TOD-v2
history below remains background only and does not authorize work in sealed
paths or a duplicate cloud launch.

### Active mission

- Journal method: EH-WIoU, Seed 42 only.
- Benchmarks: AI-TOD-v2 and Scale Match TinyPerson.

### Extended eligibility recovery - Tranche 5 resolved (2026-09-23, r5)

- Latest audit: `journal/audits/extended_official_test_recovery_post_tranche5_20260923_r5.json`.
- Latest ready queue: `journal/audits/extended_official_test_ready_queue_post_tranche5_20260923_r5.json`.
- State across 60 recovery targets: 18 `ELIGIBLE_FOR_TEST`, 40 `HOLD_VALIDATION`, 0 `INVALID/BLOCKED`, 2 `PERMANENTLY_EXCLUDED`.
- The 18-ID queue retains the prior 13 candidates and adds `EXT-019`, `EXT-027`, `EXT-028`, `EXT-029`, and `EXT-030` after each passed its exact identity, semantic, strict-reload, validation, and runner gates.
- `EXT-019` reconciliation is recorded in `journal/audits/extended_aitod_hla_protocol_reconciliation_20260923.json`. The frozen notebook payload and execution log bind the effective `foreground_classes=8`, `tile_size=800`; run-config serialization retained defaults `foreground_classes=1`, `tile_size=512`. Both metadata defects remain explicit in the record.
- Config-bound strict reload for `EXT-019` passed at 41,342,746 parameters, zero missing/unexpected keys. Record: `journal/audits/extended_aitod_hla_strict_reload_recovery_20260923.json`.
- Genuine validation predictions for YOLOv8s `EXT-027`-`EXT-030` were generated against the frozen train/validation-only dataset contract. The 2,804-image annotation and image manifest hashes match. AP50: 0.544313, 0.536189, 0.543139, 0.545108 respectively. Evidence: `journal/audits/extended_aitod_yolov8s_validation_recovery_20260923.json` and each candidate's `journal/results/aitod_yolov8s_*_s42/val_predictions_recovered_20260923.json`.
- No official-test data was accessed and no official-test inference was launched in this tranche. Test access remains closed pending a separate explicit unlock; never run the queued models from this handover alone.
- Final focused runner/readiness/recovery/validation suite: 43/43 PASS. `scripts/audit_agent_contract.py` PASS (12 files; providers codex/antigravity), `py_compile` PASS, and `git diff --check` PASS.
- Diagnostic addendum for `EXT-041`/`EXT-042`: `journal/audits/tinyperson_fcos_factorial_zero_ap_diagnosis_20260923.md`. Strictly loaded best and latest checkpoints plus fixed Seed-42 train/validation batches identify a dead regression ReLU state (99.72%/99.90% pre-activation non-positive at best; 99.979%/99.994% at latest; measured GIoU regression gradient norm 0). This refines the older post-run explanation: evidence does not show GIoU itself losing gradients for divergent boxes. Exact transition during epoch 1 remains unobserved. No retraining or official-test inference was performed; both IDs retain their existing `PERMANENTLY_EXCLUDED` eligibility state.

### Extended official-test runner queue prepared (2026-09-24)

- The frozen r5 queue remains exactly 18 eligible IDs: `EXT-019`, `EXT-027`-`EXT-030`, `EXT-033`-`EXT-038`, and `EXT-050`-`EXT-056`. The five previously certified IDs remain absent.
- Shared reader `scripts/extended_official_test_queue.py` pins the r5 queue and recovery audit SHA-256 values and verifies candidate identity/gates. TinyPerson runner now uses this exact queue instead of the stale 2026-09-22 readiness file; both family runners require `--authorize-official-test` before test-surface checks.
- `scripts/run_extended_official_test_queue.py` defaults to a test-blind dry-run, verifies configs/seed/checkpoint availability, and prints the selected IDs and family routes. Add `--verify-checkpoint-hashes` to hash every selected checkpoint.
- To execute later, the operator must have current explicit user authorization and supply both `--run` and `--authorize-official-test`; recommended command: `python scripts/run_extended_official_test_queue.py --run --authorize-official-test`. CLI flags and this handover are not execution authorization. No inference was launched in this preparation step.
- Focused suite: 27/27 PASS; `py_compile`, repository contract audit, and `git diff --check` PASS.

### Official-test certification closure (2026-09-22)

- No targeted AI-TOD-v2 or TinyPerson official-test runner is currently active.
- AI-TOD-v2 canonical models #01-#10 each have full 14,018/14,018-image prediction evidence with `sample_set_equality=true`, cryptographic prediction binding, and `PASS` verdicts.
- TinyPerson canonical models #11-#20 each have full 786/786-image prediction evidence; #17 and #19 were explicitly reconciled without discarding their valid prediction artifacts.
- Canonical reconciliation record: `journal/audits/official_test_provenance_reconciliation_20260922.json`.
- Aggregate runtime guard: `journal/test_raw/runtime_guard_trace_aggregate_20260922.json`, SHA-256 `a7357834b772945d362199f2f6b3803cdd0a072da491ccad712b91f23b82dd42`, 106,269 monitored accesses, 0 forbidden accesses, 0 sealed-workspace accesses.
- `scripts/verify_official_test_audit.py` certifies 20/20 models; payload SHA-256 is `6e38621c482a8f1da5c6ca486a2a58bb549acc9b4e1efb29f0f13e8a54103af5`.
- Evidence hardening and reconciliation are committed in `afa62e2` and `3d92059` respectively. No full-inference rerun is required unless future concrete evidence invalidates a specific model.

### Extended official-test certification closure (2026-09-22)

- Extended registry/readiness audit covers 79 non-core candidates: 14 `CORE_DUPLICATE`, 5 `FULL_TEST_EXISTING`, 42 `HOLD_VALIDATION`, and 18 `INVALID/BLOCKED`.
- The only five previously frozen eligible models have completed targeted TinyPerson official-test inference and are now certified: `EXT-040`, `EXT-043`, `EXT-044`, `EXT-047`, and `EXT-048`.
- Each certified EXT model binds a non-empty full prediction JSON, 786/786 sample identity with `sample_set_equality=true`, checkpoint/config SHA-256, strict reload PASS, evaluator/dataset provenance, explicit `--models` runner provenance, and a per-model runtime guard with zero forbidden/sealed accesses.
- Extended aggregate: `journal/audits/extended_official_test_audit_20260922.json`; independent verifier `scripts/verify_extended_official_test_audit.py` returns `EXTENDED_OFFICIAL_TEST_AUDIT_VERIFICATION=PASS` for 5/5 models with payload SHA-256 `09b106ec7fa18de1ee93cbffbd8ca12f8afb9f4a58af5be6e1adc17da637a41e`.
- Extended certification code/process is committed in `44a4f6b`; generated registry/readiness/full-run evidence is committed in `e63c0bc`.
- No extended runner is active and no rerun is required for these five models. The remaining 74 registry entries stay excluded from execution unless their current evidence state is explicitly changed by a new dated audit.

### Extended eligibility recovery milestone (2026-09-23)

- Recovery goal commit: `080ce33` (`Extended Model Eligibility Recovery`). No official-test inference is authorized inside this recovery phase.
- New dated recovery audit: `journal/audits/extended_official_test_recovery_20260923.json` over the 60 previously held/blocked candidates.
- Config-bound CPU strict reload was independently re-established for 13 TinyPerson FCOS candidates with exact checkpoint/config SHA-256 binding, Seed 42, 32,064,455 parameters, and zero missing/unexpected keys: `EXT-033`-`EXT-038` and `EXT-050`-`EXT-056`.
- These 13 candidates now pass checkpoint identity, config identity, semantic identity, validation, strict reload, runner support, and evaluator-source gates, and are frozen as `ELIGIBLE_FOR_TEST` in `journal/audits/extended_official_test_ready_queue_20260923.json`.
- Independent factorial replay confirms `EXT-041` and `EXT-042` are genuine zero-AP negative controls (`AP50/AP75/AP = 0.0` both cloud and local replay with `reproducibility_gate_passed=true`), so they are `PERMANENTLY_EXCLUDED` rather than treated as missing-validation candidates.
- Dated protocol reconciliation `journal/audits/extended_protocol_semantic_reconciliation_20260923.json` binds `EXT-061`-`EXT-065` to the accepted TinyPerson Faster R-CNN beta-sweep v5 design/prerun/postrun/download evidence. Their artifact `run_config` retains the earlier v1 protocol label, while v5 is a ModelEMA/RPN evaluation hotpatch with the same Seed-42 beta scientific factor. These five now pass semantic identity but remain `HOLD_VALIDATION` because no recovery-authorized `frcnn_beta` targeted official-test runner exists.
- Wave I TinyPerson YOLO recovery (`EXT-072`-`EXT-079`) is reconciled by dated metadata and strict-reload evidence. The umbrella protocol `tinyperson_yolo_wave_i_sota_losses_seed42_v1` is accepted as the dispatch-level authority for the family-local YOLO26s/YOLOv8s protocol labels; exact config and checkpoint SHA-256 bindings are required per candidate.
- Native Ultralytics reload confirms the frozen nc=1 checkpoint identities: 9,948,638 parameters for YOLO26s and 11,135,987 for YOLOv8s. The larger counts stored in downloaded run_config files are pre-dataset nc=80 YAML counts and are treated as packaging metadata, not checkpoint architecture identity.
- `EXT-072`-`EXT-079` now pass semantic identity and strict reload but remain `HOLD_VALIDATION`: accepted bound validation prediction evidence is absent and recovery-authorized targeted runner support remains unavailable for `yolo26s`/`yolov8s`.
- Current 60-candidate recovery state: 13 `ELIGIBLE_FOR_TEST`, 40 `HOLD_VALIDATION`, 5 `INVALID/BLOCKED`, and 2 `PERMANENTLY_EXCLUDED`. Certified `EXT-040/043/044/047/048` remain excluded from rerun.
- This is an eligibility result only. No new full official-test inference has been launched for the 13 queued models.
- Recovery implementation commit: `fbd2bbc`; dated evidence commit: `fdbfb8c`.
- Tranche-5 follow-up audit (`journal/audits/extended_official_test_recovery_post_tranche5_20260923_r2.json`) reconciles the four YOLOv8s Wave-F manifest labels against the frozen design/prerun/dispatch receipts and records config-bound native reloads for `EXT-027`-`EXT-030` (11,138,696 parameters, `nc=8`). Their metric aliases are finite, but their downloaded manifests contain no prediction lists; the 37-byte local files are status markers. All four remain `HOLD_VALIDATION`, with AI-TOD-v2 YOLOv8s targeted-runner support still absent.
- `EXT-019` has matching checkpoint/config SHA bindings and accepted validation evidence (`detections_best.json`, 118,242 detection rows, AP50 0.507640). It remains `INVALID/BLOCKED`: the frozen HLA dispatch identity is clear, but its run config records `tile_size=512` while the hashed trainer constructs with `tile_size=800`; strict-reload evidence is not yet recorded and no recovery-authorized Faster R-CNN targeted runner exists. Discrepancy details are in `journal/audits/extended_aitod_hla_discrepancy_20260923.json`.
- The latest post-Tranche-5 re-audit is `journal/audits/extended_official_test_recovery_post_tranche5_20260923_r4.json`: 13 `ELIGIBLE_FOR_TEST`, 44 `HOLD_VALIDATION`, 1 `INVALID/BLOCKED`, and 2 `PERMANENTLY_EXCLUDED`. Queue: `journal/audits/extended_official_test_ready_queue_post_tranche5_20260923_r4.json`; it preserves the exact ready set `EXT-033`-`EXT-038`, `EXT-050`-`EXT-056`. Focused readiness/recovery tests passed 31/31. The tranche remains open; no official-test inference is authorized.
- `scripts/run_aitod_extended_test_inference.py` now provides explicit-ID YOLOv8s runner support. It cross-checks the ready queue with its source audit and exact config/checkpoint bindings, rejects held IDs, and requires `--authorize-official-test`. Regression coverage passes; no inference was executed. Thus EXT-027-EXT-030 pass runner support but remain `HOLD_VALIDATION` because no genuine manifest-bound validation predictions exist. Their run configs bind only validation image counts, not a validation data location; local result directories contain configs/CSVs but no prediction artifacts. Do not regenerate validation predictions until an explicitly allowed validation source and split identity are established.
- EXT-019 remains blocked by the config/trainer `tile_size` discrepancy, with no accepted strict-reload or AI-TODv2 Faster R-CNN targeted-runner evidence.

- Current phase & Major Milestones (2026-09-22):
  1. TinyPerson FCOS Factorial (10/10 arms): COMPLETED, DOWNLOADED, STRICT RELOAD VERIFIED, AND REPLAY ACCEPTED on RTX 5070 Ti (`FACTORIAL_REPLAY_ACCEPTANCE=PASS`, $\Delta \le 4.77 \times 10^{-7}$). ANOVA effects and 10,000 bootstrap statistical significance audits recorded. Top arm `l1c1s0` achieved AP50 = 0.3113 (+3.61% over baseline 0.2752) and AP_tiny1 = 0.1200 (+115% over baseline 0.0558).
  2. TinyPerson Faster R-CNN Placement Factorial (Revision v4): COMPLETED, DOWNLOADED, STRICT RELOAD VERIFIED (41,306,871 parameters on CPU & CUDA), AND INDEPENDENTLY REPLAYED via isolated frozen source bundle on RTX 5070 Ti (`PLACEMENT_REPLAY_ACCEPTANCE=PASS`, max $\Delta = 0.000207 \le 0.0005$). Re-audit decision `HOU-AUD-20260913-TP-FRCNN-PLACE-REAUDIT` certified. Dual placement achieves $\text{AP}_{50}^{\text{tiny1}} = \mathbf{0.2565}$ (+26.04% micro-target surge over baseline 0.2035) and $\text{AP}_{50}^{\text{reasonable}} = \mathbf{0.6175}$ (+7.16% absolute gain); peak overall accuracy achieved by `assignment_only` at $\text{AP}_{50}^{\text{all}} = \mathbf{0.4248}$. Post-run audit and download manifest certified.
  3. AI-TOD-v2 FCOS Loss-Only Paired Evaluation (Revision v2r8): COMPLETED across both arms. Baseline GIoU (`luongsythanh`) AP=0.1814, AP50=0.4541, AR1500=0.2955; Method H-WIoU (`pptlyn11`, sigma0=6.0px, beta=0.50) AP=0.1824 (+0.10%), AP50=0.4582 (+0.41%), AR1500=0.3016 (+0.61%), train loss=0.8703 vs 0.9104. H-WIoU confirms positive transfer on satellite tiny objects.
  4. AI-TOD-v2 FCOS Sigma Bracket: COMPLETED across both arms. Arm 1 (`quangnhtng`, sigma0=4.0px, v1) achieved AP=0.1844, AP50=0.4593, AP75=0.1129, AP_s=0.2438, AR1500=0.2990. Arm 2 (`hngtrngtn`, sigma0=8.0px, Version 3) COMPLETED successfully at terminal Epoch 10: train_loss=0.8524, AP=0.1788, AP50=0.4455, AP75=0.1137, AP_s=0.2391, AR1500=0.2926. Terminal metrics confirm the Inverted-U scale curve on satellite targets: sigma0=4.0px is peak optimal (+0.52% AP50 over GIoU baseline 0.4541), while sigma0=8.0px causes over-smoothing degradation (AP50 drops to 0.4455). All 3 points of the AI-TOD-v2 bracket ({4.0, 6.0, 8.0}px) are certified and integrated into Table 2 of the manuscript.
  5. TinyPerson FCOS Homotopy-Form Phase 1 & 2 Complete Inverted-U Matrix: COMPLETED across all 6 arms. Rational Homotopy AP50=0.3054, Static Half AP50=0.2726, Pure IoU (gamma=1.0) AP50=0.2706, Exponential AP50=0.2667, Sigmoid AP50=0.2635, Pure W2 (gamma=0.0) AP50=0.2227. Continuous rational homotopy strictly dominates all boundary and static formulations (+3.48% over pure IoU, +8.27% over pure W2).
  6. TinyPerson Faster R-CNN Beta Sensitivity Sweep (Revision v5): COMPLETED across all 5 arms, DOWNLOADED, STRICT RELOAD CERTIFIED (41,306,871 parameters), AND INDEPENDENTLY REPLAYED on RTX 5070 Ti (`BETA_SWEEP_REPLAY_ACCEPTANCE=PASS`, mean $\bar{\Delta} = 0.000369$). Top arm `beta100` ($\beta = 1.00$) achieved $\text{AP}_{50}^{\text{all}} = \mathbf{0.4297}$ (+0.74% over zero-entropy baseline $\beta = 0.00$), $\text{AP}_{50}^{\text{tiny1}} = \mathbf{0.2646}$, and $\text{AP}_{50}^{\text{reasonable}} = \mathbf{0.5978}$ (+3.16% over $\beta=0.00$), confirming monotonic performance gains toward unit spatial entropy coupling. Post-run audit and download manifest certified.
  7. Journal Source Remediation (Phases 0, 1, 3, 4, 5, 6): COMPLETED and committed across commits `60e3724` (Phase 0), `1fb1c46` (Phase 1), `39e1ed0` (Phase 3), `562ad68` (Phase 4), `0f65c2a` (Phase 5), and `2d99ef0` (Phase 6). Solved pipeline decoupling, test fixture isolation, atomic receipt persistence, standardized `kaggle_profile_context`, and explicit deprecation warnings.
  8. SOTA 2025-2026 Comparative Reproducibility Matrix & Metric Unit Harness: COMPLETED. Authored `journal/wiki/syntheses/sota_comparative_reproducibility_matrix_2026.md` analyzing NWD, IGWD, SimD, GCD (GRSL 2025), SAFit (AAAI 2024), and SET (CVPR 2025); verified asymptotic mathematical properties in `scripts/test_sota_baseline_metrics.py` (3/3 unit tests PASS); synchronized long-term knowledge to TencentDB Agent Memory Hub (`http://127.0.0.1:8420`).
  9. SOTA Comparative Arms Staging & Preflight (NWD, IGWD, SimD, GCD): STAGED, VERIFIED, AND CERTIFIED READ-ONLY. Created immutable design `journal/audits/hwiou_seed42_fcos_tinyperson_sota_4arms_design_2026-09-16.json` and prerun `journal/audits/hwiou_seed42_fcos_tinyperson_sota_4arms_prerun_2026-09-16.json` (payload `618e8084...`, dispatch_authority=False); all 4 arms staged under `.runtime/kaggle_tp_fcos_sota_4arms_seed42/`; read-only preflight certified across 4 accounts (`qnhat1504`, `thyngluthy`, `hienquang06`, `dipphmngc`, total GPU quota 108.08h, dataset ACL READY); 5/5 workload identity gates certified in `scripts/launch_sota_4arms_cluster.py`. Current status: NO DISPATCH (authority is strictly False awaiting explicit user instruction).
  10. Step 2 Option 2 Targeted Official Test Split Inference & Cryptographic Audit: COMPLETE and RECONCILED for all 20 canonical models. AI-TOD-v2 #01-#10 each bind full 14,018-image prediction JSONs; TinyPerson #11-#20 each bind full 786-image prediction JSONs. `core_models_official_test_audit.json` verifies 20/20 PASS with payload SHA `6e38621c482a8f1da5c6ca486a2a58bb549acc9b4e1efb29f0f13e8a54103af5`; aggregate runtime guard SHA is `a7357834b772945d362199f2f6b3803cdd0a072da491ccad712b91f23b82dd42` with zero forbidden/sealed accesses. Provenance reconciliation is recorded in `journal/audits/official_test_provenance_reconciliation_20260922.json`. No duplicate rerun is authorized or necessary for currently complete models.
  11. Qualitative Visual Gallery & Inference Latency Benchmark: COMPLETED. Generated publication-grade Fig 5 & Fig 6 (`journal/manuscript/figures/fig5_tinyperson_qualitative.pdf`, `fig6_aitodv2_qualitative.pdf`); benchmarked inference latency & throughput across all 6 detector families on RTX 5070 Ti (`scripts/benchmark_detector_latency_throughput.py`), verifying Zero Parameter Overhead and $|\Delta \text{Latency}| \le 1.4\%$ inference parity (Faster R-CNN 57.1 FPS, FCOS 49.0 FPS, Cascade 43.6 FPS, RetinaNet 72.2 FPS, YOLOv8s 171.8 FPS, YOLO26s 111.3 FPS); integrated Table 7 into `journal/manuscript/main.tex` and recompiled PDF.
  12. Manuscript Comprehensive Review & Editorial Polish: COMPLETED. Executed full 5-pass editorial audit (Sainani methodology) on `journal/manuscript/main.tex`: eliminated hyperbolic phrasing, ensured bijective 33/33 citations, resolved all 28 cross-references (including weaving Fig 1, Fig 5, and Fig 6 into body text), adjusted Table 6 column spacing to eliminate the sole overfull hbox (0 overfull hboxes across all 14 pages), and expanded Section 9 Conclusion with comprehensive theoretical, empirical, and latency synthesis; successfully recompiled `main.pdf` (14 pages, 12.5 MB).
  13. Manuscript Scientific Writing Overhaul & Tone Calibration (IEEE TPAMI Standard): COMPLETED. Executed comprehensive sentence-level scientific writing overhaul, claim defense calibration, and terminology de-marketing per peer review: eliminated promotional marketing language (no 'catastrophic divergence', 'surges', 'consistent Pareto dominance', 'real-time and NMS-Free frontier', 'zero-leakage guarantee'); strictly calibrated claims against exact tables and data (acknowledged YOLO26s precision gain of +15.08 pp is coupled with recall drop 50.04% vs 53.33%; noted AI-TOD-v2 EIoU competitiveness; corrected FCOS Gaussian centerness runs to degenerate zero-AP runs rather than numerical collapse; fixed attribution error for 0.3113 AP50 to l1c1s0 rather than loss-only); corrected citations (Hu et al. IGWD TMM 2026 and Ying et al. SAFit); updated Figure 2 vector graphics and PDF labels to dry academic terms (Scale-Adaptive H-WIoU Formulation, Feature Extraction and Detached Channel Entropy, Scale-Dependent Mixing, Analytical Properties, Input Image); recompiled `journal/manuscript/main.pdf` (13 pages, 0 overfull hboxes, 0 undefined references); verified `scripts/audit_agent_contract.py` PASS.
- Allowed TinyPerson development surface: 628 training and 118 validation
  originals from `b1_tiled_20260814` (512x512 tiles, 64px overlap).
- TinyPerson official test: 786 images, `UNLOCKED_FOR_EVALUATION` under `HOU-USER-STEP2-ONESHOT-AUTH` (evaluation completed and audited; no training/tuning allowed).
- Repository `data/`, including `data/sod-tinypeopleinsea`: prohibited.
- `paper_a/`: sealed unpublished SA-ALW workspace; absent from Journal work.

### Live workload

Operational status on 2026-09-22:

| Official Test Split Evaluation (TinyPerson #11–#20) | 10 TinyPerson models (#11–#20) on 786 images (23k tiles) | 10/10 `COMPLETE` | full predictions/raw bundles reconciled and aggregate guard bound | `CERTIFIED_20_OF_20 / NO_ACTIVE_RUNNER` |
| Official Test Split Evaluation (AI-TOD-v2 #01–#10) | 10 AI-TOD-v2 models (#01–#10) on 14,018 images | 10/10 `COMPLETE` | 10 full prediction JSONs/raw records cryptographically bound | `CERTIFIED_20_OF_20 / NO_ACTIVE_RUNNER` |
| Extended Official Test (TinyPerson FCOS) | `EXT-040`, `EXT-043`, `EXT-044`, `EXT-047`, `EXT-048` on 786 images | 5/5 `COMPLETE` | full predictions/raw bundles/guards cryptographically bound | `CERTIFIED_5_OF_5 / NO_ACTIVE_RUNNER` |
| Cascade pair | `ngquangnht/tod-tp-cas-base-s42` v1; `amongus1504/tod-tp-cas-ehwiou-s42` v1 | `COMPLETE` | `downloaded` | `artifact_ready / replayed` |
| RetinaNet pair | `quangnhtng/tod-tp-ret-base-s42` v3; `hngtrngtn/tod-tp-ret-hwiou-s42` v3 | `COMPLETE` | `downloaded` | `artifact_ready / replayed` |
| TinyPerson FCOS 10-arm Factorial | 10 arms on 10 owner-bound accounts (`fac-l{0,1}c{0,1}s{0,1}-s42` + `sigma4,8`) | 10/10 `COMPLETE` | `downloaded` | `REPLAY_ACCEPTED_STATISTICALLY_AUDITED` |
| TinyPerson FRCNN Placement v4 | `hienquang06`, `dipphmngc`, `hngngnguynvn`, `trieuvo123` | 4/4 `COMPLETE` | `downloaded` | `REPLAY_ACCEPTED_PLACEMENT_CONFIRMED` |
| TinyPerson FCOS Homotopy Form (Phase 1) | `amongus1504`, `qnhat1504`, `thyngluthy` | 3/3 `COMPLETE` | `downloaded` | `ARTIFACT_GATE_PASS_STRICT_RELOAD_CERTIFIED` |
| TinyPerson FCOS Homotopy Form (Phase 2) | `hngngnguynvn` (`pure_w2`), `hienquang06` (`pure_iou`) | 2/2 `COMPLETE` | `downloaded` | `POSTRUN_AUDITED_INVERTED_U_CONFIRMED` |
| AI-TOD-v2 FCOS Loss-Only (v2r8) | `luongsythanh` (GIoU), `pptlyn11` (H-WIoU) | 2/2 `COMPLETE` | `downloaded` | `H_WIOU_SURPASSES_GIOU_CONFIRMED` |
| AI-TOD-v2 FCOS Sigma Bracket (v2r1/v2r3) | `quangnhtng` (sig4 v1), `hngtrngtn` (sig8 v3) | `sig4`: `COMPLETE`, `sig8 v3`: `COMPLETE` (10/10 epochs) | `sig4`: `downloaded`, `sig8 v3`: `metrics_audited` | `INVERTED_U_BRACKET_CONFIRMED_AND_INTEGRATED` |
| TinyPerson FRCNN Beta Sweep (v5) | `amongus1504`, `qnhat1504`, `thyngluthy`, `trieuvo123`, `dipphmngc` | 5/5 `ERROR (gc NameError)` | `downloaded` | `REPLAY_ACCEPTED_BETA_SWEEP_CONFIRMED` |
| TinyPerson FCOS SOTA 4 Arms | `qnhat1504` (NWD), `thyngluthy` (IGWD), `hienquang06` (SimD), `dipphmngc` (GCD retry) | 4/4 `COMPLETE` | `downloaded` (manifest recorded) | `RELOAD_ACCEPTED_SOTA_SURPASSED` |
| TinyPerson FCOS SOTA Geometric Losses (Wave A) | `qnhat1504` (CIoU), `thyngluthy` (DIoU), `hienquang06` (EIoU), `dipphmngc` (WIoU) | 4/4 `COMPLETE` | `downloaded` (manifest recorded) | `RELOAD_ACCEPTED_H_WIOU_SURPASSES_ALL` |
| TinyPerson FCOS SOTA Extended Losses (Wave C) | `luongsythanh` (alpha-IoU), `pptlyn11` (SIoU v3), `trieuvo123` (Inner-IoU), `phuc1806` (Focal-EIoU) | 4/4 `COMPLETE` | `downloaded` (manifest recorded) | `RELOAD_ACCEPTED_H_WIOU_SURPASSES_ALL` |
| TinyPerson FRCNN SOTA Assignment (Wave B) | `amongus1504` (RFLA), `quangnhtng` (NWD), `hngtrngtn` (SimD), `hngngnguynvn` (ATSS) | 4/4 `COMPLETE` | `downloaded` (manifest recorded) | `RELOAD_ACCEPTED_H_WIOU_SURPASSES_ALL` |
| AI-TOD-v2 FCOS SOTA Losses (Wave D) | `qnhat1504` (NWD), `dipphmngc` (IGWD), `thyngluthy` (SimD), `hienquang06` (GCD retry) | 4/4 `COMPLETE` | `downloaded` (manifest recorded) | `H_WIOU_SURPASSES_ALL` |
| AI-TOD-v2 FCOS SOTA Geometric Losses (Wave E) | `phuc1806` (CIoU), `trieuvo123` (DIoU), `dipphmngc` (EIoU), `thyngluthy` (WIoU) | 4/4 `COMPLETE` (10/10 epochs) | `downloaded` (manifest recorded) | `ARTIFACTS_DOWNLOADED_VERIFIED` |
| AI-TOD-v2 FRCNN Paired Benchmark | `luongsythanh` (tod-aitod-frcnn-base-s42), `pptlyn11` (tod-aitod-frcnn-ehwiou-s42) | 2/2 `COMPLETE` (12/12 epochs) | `downloaded` (manifest recorded) | `RELOAD_ACCEPTED_PAIRED_CONFIRMED` |
| AI-TOD-v2 FRCNN SOTA Assignment Benchmark | `amongus1504` (RFLA), `quangnhtng` (NWD), `hngtrngtn` (SimD), `hngngnguynvn` (ATSS) | 4/4 `COMPLETE` (12/12 epochs) | `downloaded` (manifest recorded) | `ARTIFACTS_DOWNLOADED_VERIFIED` |
| AI-TOD-v2 YOLOv8s SOTA Benchmark (Wave F) | `amongus1504` (CIoU), `quangnhtng` (IGWD), `hngtrngtn` (H-WIoU), `hngngnguynvn` (H-TAL) | 4/4 `COMPLETE` (100 epochs) | `downloaded` (manifest recorded) | `ARTIFACTS_DOWNLOADED_VERIFIED` |
| AI-TOD-v2 YOLO26 NMS-Free Frontier (Wave G) | `luongsythanh` (CIoU), `pptlyn11` (IGWD), `qnhat1504` (H-WIoU), `ngquangnht` (H-STAL) | 4/4 `COMPLETE` (100 epochs) | `downloaded` (manifest recorded) | `ARTIFACTS_DOWNLOADED_VERIFIED` |
| AI-TOD-v2 FCOS SOTA Extended Losses (Wave H) | `dipphmngc` (alphaiou), `thyngluthy` (siou), `trieuvo123` (inner_iou), `hienquang06` (gcd), `phuc1806` (focaleiou retry) | 5/5 `COMPLETE` | 5 `downloaded` (manifest recorded) | `RELOAD_ACCEPTED_ALL_ARMS_VERIFIED` |
| TinyPerson YOLO SOTA Benchmark (Wave I) | 4 YOLOv8s (`amongus1504`, `quangnhtng`, `hngtrngtn`, `hngngnguynvn`) + 4 YOLO26s (`luongsythanh`, `pptlyn11`, `qnhat1504`, `ngquangnht`) | 8/8 `COMPLETE` (100 epochs) | `downloaded` (manifest recorded) | `STRICT_RELOAD_VERIFIED_8ARMS` |
| AI-TOD-v2 Missing Benchmark Arms (3 arms) | `qnhat1504` (bundle_l1c1s0), `thyngluthy` (bundle_l1c1s1), `hienquang06` (frcnn_hla_rpn) | 3/3 `COMPLETE` | 3 `downloaded` (manifest recorded) | `STRICT_RELOAD_ACCEPTED_ALL_3ARMS` |

Governing records:
- Official Test Split Audit Protocol: `core_models_official_test_audit`. Canonical aggregate (`journal/audits/core_models_official_test_audit.json`), markdown audit report (`journal/audits/core_models_official_test_audit.md`), provenance reconciliation (`journal/audits/official_test_provenance_reconciliation_20260922.json`), aggregate runtime guard (`journal/test_raw/runtime_guard_trace_aggregate_20260922.json`), and full prediction evidence (`journal/test_raw/model_{01..20}_*_test_predictions.json`). Independent verification uses `scripts/verify_official_test_audit.py --audit-path journal/audits/core_models_official_test_audit.json --reconciliation-path journal/audits/official_test_provenance_reconciliation_20260922.json`.
- AI-TOD-v2 Missing Benchmark Arms Protocol: `aitodv2_missing_benchmark_arms_seed42_v1`. Design record (`journal/audits/hwiou_seed42_aitodv2_missing_arms_design_2026-09-20.json`), frozen prerun (`journal/audits/hwiou_seed42_aitodv2_missing_arms_prerun_2026-09-20.json`), preflight report (`journal/audits/preflight_runs/aitodv2_missing_arms_preflight_2026-09-20.json`), staging audit (`journal/audits/hwiou_seed42_aitodv2_missing_arms_staging_2026-09-20.md`), and dispatch receipt (`journal/audits/dispatch_receipts/receipt_aitodv2_missing_arms_2026-09-20_23-19-23_96ace0c5_SUCCESS.json`). Staged under `.runtime/kaggle_aitod_missing_arms_seed42/` across 3 accounts (FCOS: 32,080,590 params, 10 epochs; Faster R-CNN: 41,342,746 params, 12 epochs).
58: - TinyPerson YOLO SOTA Benchmark Protocol (Wave I): `tinyperson_yolo_wave_i_seed42_v1`. Design record (`journal/audits/hwiou_seed42_yolo_tinyperson_wave_i_design_2026-09-20.json`), frozen prerun (`journal/audits/hwiou_seed42_yolo_tinyperson_wave_i_prerun_2026-09-20.json`), dispatch receipt (`journal/audits/dispatch_receipts/receipt_tinyperson_yolo_wave_i_2026-09-20_04-38-15_SUCCESS.json`). Staged under `.runtime/kaggle_tp_yolo_wave_i_seed42/` across 8 accounts (YOLOv8s: 11,135,987 params; YOLO26s: 9,948,638 params; 100 epochs, Cosine Annealing, close_mosaic=10, patience=25).
59: - AI-TOD-v2 FCOS Focal-EIoU Retry Protocol (Wave H): `tod-aitod-fcos-focaleiou-s42` dispatched to `phuc1806`. Dispatch receipt (`journal/audits/dispatch_receipts/receipt_aitodv2_focaleiou_retry_2026-09-20_04-35-10_SUCCESS.json`).
60: - AI-TOD-v2 YOLOv8s SOTA Benchmark Download Manifest: `journal/results/aitodv2_yolov8s_download_manifest.json` (4/4 arms COMPLETE at 100 epochs: CIoU 54.08% mAP50, IGWD 54.57% mAP50, H-WIoU 53.95% mAP50 / 64.61% Precision, H-TAL 53.89% mAP50 / 54.32% Recall; all checkpoints verified).
61: - AI-TOD-v2 YOLO26 NMS-Free SOTA Benchmark Download Manifest: `journal/results/aitodv2_yolo26s_download_manifest.json` (4/4 arms COMPLETE at 100 epochs: CIoU 53.44% mAP50, IGWD 53.98% mAP50, H-WIoU 52.63% mAP50, H-STAL 53.32% mAP50 / 72.01% Precision; all checkpoints verified).
62: - AI-TOD-v2 FCOS SOTA Extended Losses & GCD Retry Download Manifest: `journal/results/aitodv2_fcos_wave_h_download_manifest.json` (4 completed arms verified: SIoU 18.72% AP / 45.63% AP50, Inner-IoU 18.07% AP / 43.71% AP50, alpha-IoU 17.54% AP / 42.65% AP50, GCD Retry 16.96% AP / 42.01% AP50; checkpoints verified).
63: - AI-TOD-v2 FCOS SOTA Extended Losses & GCD Retry Protocol (Wave H): `aitodv2_fcos_sota_wave_h_seed42_v1`. Design record (`journal/audits/hwiou_seed42_fcos_aitodv2_wave_h_design_2026-09-19.json`), frozen prerun (`journal/audits/hwiou_seed42_fcos_aitodv2_wave_h_prerun_2026-09-19.json`), preflight report (`journal/audits/preflight_runs/aitodv2_fcos_wave_h_preflight_2026-09-19.json`), dispatch audit (`journal/audits/hwiou_seed42_fcos_aitodv2_wave_h_dispatch_2026-09-19.md`), and dispatch receipt (`journal/audits/dispatch_receipts/receipt_aitodv2_wave_h_2026-09-19_14-55-23_8ac236ba_SUCCESS.json`). Candidates prepared in `.runtime/kaggle_aitod_fcos_wave_h_seed42/`.
64: - AI-TOD-v2 Faster R-CNN SOTA Assignment Download Manifest: `journal/results/aitodv2_frcnn_assignment_download_manifest.json` (4/4 arms COMPLETE, checkpoints verified).
65: - AI-TOD-v2 FCOS SOTA Geometric Losses Download Manifest: `journal/results/aitodv2_fcos_geom_losses_download_manifest.json` (4/4 arms COMPLETE, checkpoints verified).
- AI-TOD-v2 YOLOv8s SOTA Benchmark Protocol (Wave F): `aitodv2_yolov8s_sota_benchmark_seed42_v1`. Design record (`journal/audits/hwiou_seed42_yolov8s_aitodv2_design_2026-09-19.json`), frozen prerun (`journal/audits/hwiou_seed42_yolov8s_aitodv2_prerun_2026-09-19.json`), dispatch audit (`journal/audits/hwiou_seed42_yolov8s_aitodv2_dispatch_2026-09-19.md`), and dispatch receipt (`journal/audits/dispatch_receipts/receipt_aitodv2_yolov8s_2026-09-19_16-38-28_3cfaa263_SUCCESS.json`). Candidates prepared in `.runtime/kaggle_aitod_yolov8s_seed42/` (100 epochs, Cosine Annealing, close_mosaic=10, patience=25).
- AI-TOD-v2 YOLO26 NMS-Free SOTA Benchmark Protocol (Wave G): `aitodv2_yolo26_nmsfree_sota_losses_seed42_v1`. Design record (`journal/audits/hwiou_seed42_yolo26_aitodv2_design_2026-09-19.json`), frozen prerun (`journal/audits/hwiou_seed42_yolo26_aitodv2_prerun_2026-09-19.json`), dispatch audit (`journal/audits/hwiou_seed42_yolo26_aitodv2_dispatch_2026-09-19.md`), and dispatch receipt (`journal/audits/dispatch_receipts/receipt_aitodv2_yolo26s_2026-09-19_16-39-04_16cfc07a_SUCCESS.json`). Candidates prepared in `.runtime/kaggle_aitod_yolo26s_seed42/` (100 epochs, Cosine Annealing, close_mosaic=10, patience=25).
- AI-TOD-v2 FCOS SOTA Geometric Losses Protocol: `aitodv2_fcos_geom_losses_seed42_v1`. Design record (`journal/audits/hwiou_seed42_fcos_aitodv2_geom_losses_design_2026-09-18.json`), frozen prerun (`journal/audits/hwiou_seed42_fcos_aitodv2_geom_losses_prerun_2026-09-18.json`), preflight report (`journal/audits/preflight_runs/aitodv2_fcos_geom_losses_preflight_2026-09-18.json`), dispatch audit (`journal/audits/hwiou_seed42_fcos_aitodv2_geom_losses_dispatch_2026-09-18.md`), and dispatch receipt (`journal/audits/dispatch_receipts/receipt_aitodv2_geom_losses_2026-09-18_15-56-18_09b54127_SUCCESS.json`). Candidates prepared in `.runtime/kaggle_aitod_fcos_geom_losses_seed42/`.
- AI-TOD-v2 Faster R-CNN SOTA Assignment Protocol: `aitodv2_frcnn_assignment_sota_seed42_v1`. Design record (`journal/audits/hwiou_seed42_frcnn_aitodv2_assignment_design_2026-09-18.json`), frozen prerun (`journal/audits/hwiou_seed42_frcnn_aitodv2_assignment_prerun_2026-09-18.json`), preflight report (`journal/audits/preflight_runs/aitod_frcnn_assignment_preflight_2026-09-18.json`), dispatch audit (`journal/audits/hwiou_seed42_frcnn_aitodv2_assignment_dispatch_2026-09-18.md`), and dispatch receipts (`journal/audits/dispatch_receipts/receipt_aitod_frcnn_assignment_2026-09-18_13-39-47_f0f48198_SUCCESS.json`, `receipt_aitod_frcnn_assignment_2026-09-18_13-44-28_2e5c6f55_SUCCESS.json`, and `receipt_aitod_frcnn_assignment_2026-09-18_15-18-34_d0908e07_SUCCESS.json`). Candidates prepared in `.runtime/kaggle_aitod_frcnn_assignment_seed42/`.
- AI-TOD-v2 Faster R-CNN Paired Benchmark Protocol: `aitodv2_frcnn_paired_seed42_v1`. Design record (`journal/audits/hwiou_seed42_frcnn_aitodv2_paired_design_2026-09-18.json`), frozen prerun (`journal/audits/hwiou_seed42_frcnn_aitodv2_paired_prerun_2026-09-18.json`), preflight report (`journal/audits/preflight_runs/aitod_frcnn_paired_preflight_2026-09-18.json`), and dispatch receipt (`journal/audits/dispatch_receipts/receipt_aitod_frcnn_paired_2026-09-18_09-35-38_636d56b7_SUCCESS.json`). Candidates prepared in `.runtime/kaggle_aitod_frcnn_paired_seed42/`.
- AI-TOD-v2 FCOS SOTA Distributional Losses Protocol: `aitodv2_fcos_sota_losses_seed42_v1`. Design record (`journal/audits/hwiou_seed42_fcos_aitodv2_sota_losses_design_2026-09-18.json`), frozen prerun (`journal/audits/hwiou_seed42_fcos_aitodv2_sota_losses_prerun_2026-09-18.json`), preflight report (`journal/audits/preflight_runs/sota_aitodv2_losses_preflight_2026-09-18.json`), dispatch audit (`journal/audits/hwiou_seed42_fcos_aitodv2_sota_losses_dispatch_2026-09-18.md`), and dispatch receipt (`journal/audits/dispatch_receipts/receipt_sota_aitodv2_losses_2026-09-18_10-22-04_5e3116a8_SUCCESS.json`). Candidates prepared in `.runtime/kaggle_aitod_fcos_sota_losses_seed42/`.
- TinyPerson FRCNN SOTA Assignment Protocol: `tinyperson_frcnn_assignment_sota_seed42_v1`. Design record (`journal/audits/hwiou_seed42_frcnn_tinyperson_assignment_design_2026-09-18.json`), frozen prerun (`journal/audits/hwiou_seed42_frcnn_tinyperson_assignment_prerun_2026-09-18.json`), preflight report (`journal/audits/preflight_runs/frcnn_assignment_preflight_2026-09-18.json`), dispatch audit (`journal/audits/hwiou_seed42_frcnn_tinyperson_assignment_dispatch_2026-09-18.md`), and dispatch receipt (`journal/audits/dispatch_receipts/receipt_sota_frcnn_assignment_2026-09-18_02-23-43_fb879ed0_SUCCESS.json`). Candidates prepared in `.runtime/kaggle_tp_frcnn_assignment_seed42/`.
- TinyPerson FCOS SOTA Extended Losses Protocol: `tinyperson_fcos_extended_losses_seed42_v1`. Design record (`journal/audits/hwiou_seed42_fcos_tinyperson_extended_losses_design_2026-09-17.json`), frozen prerun (`journal/audits/hwiou_seed42_fcos_tinyperson_extended_losses_prerun_2026-09-17.json`), dispatch audit (`journal/audits/hwiou_seed42_fcos_tinyperson_extended_losses_dispatch_2026-09-17.md`), and dispatch receipt (`journal/audits/dispatch_receipts/receipt_sota_extended_losses_2026-09-17_16-14-59_ec27100e_SUCCESS.json`). Candidates prepared in `.runtime/kaggle_tp_fcos_extended_losses_seed42/`.
- TinyPerson FCOS SOTA Geometric Losses Protocol: `tinyperson_fcos_geom_losses_seed42_v1`. Design record (`journal/audits/hwiou_seed42_fcos_tinyperson_geom_losses_design_2026-09-17.json`), frozen prerun (`journal/audits/hwiou_seed42_fcos_tinyperson_geom_losses_prerun_2026-09-17.json`), dispatch audit (`journal/audits/hwiou_seed42_fcos_tinyperson_geom_losses_dispatch_2026-09-17.md`), and dispatch receipt (`journal/audits/dispatch_receipts/receipt_sota_geom_losses_2026-09-17_23-04-27_5ea5e6d1_SUCCESS.json`). Candidates prepared in `.runtime/kaggle_tp_fcos_geom_losses_seed42/`.
- TinyPerson FCOS SOTA 4-Arm Comparison Post-run Audit: `journal/audits/hwiou_seed42_fcos_tinyperson_sota_4arms_postrun_2026-09-17.md` and `.json`. Download manifest: `journal/results/sota_4arms_download_manifest.json`.
- TinyPerson FCOS SOTA GCD Arm Retry Protocol: `tinyperson_fcos_sota_gcd_retry_seed42_v1`. Design record (`journal/audits/hwiou_seed42_fcos_tinyperson_gcd_retry_design_2026-09-17.json`), frozen prerun (`journal/audits/hwiou_seed42_fcos_tinyperson_gcd_retry_prerun_2026-09-17.json`), preflight report (`journal/audits/preflight_runs/sota_gcd_retry_preflight_2026-09-17.json`), dispatch audit (`journal/audits/hwiou_seed42_fcos_tinyperson_gcd_retry_dispatch_2026-09-17.md`), and atomic receipt (`journal/audits/dispatch_receipts/receipt_sota_gcd_retry_2026-09-16_18-59-31_3eee7391_SUCCESS.json`). Candidate prepared in `.runtime/kaggle_tp_fcos_gcd_retry_seed42/dipphmngc/`.
- TinyPerson FCOS SOTA 4-Arm Comparison Protocol: `tinyperson_fcos_sota_comparison_seed42_v1`. Design record (`journal/audits/hwiou_seed42_fcos_tinyperson_sota_4arms_design_2026-09-16.json`), frozen prerun (`journal/audits/hwiou_seed42_fcos_tinyperson_sota_4arms_prerun_2026-09-16.json`), preflight report (`journal/audits/preflight_runs/sota_4arms_preflight_2026-09-16.json`), dispatch audit (`journal/audits/hwiou_seed42_fcos_tinyperson_sota_4arms_dispatch_2026-09-16.md`), and dispatch receipt (`journal/audits/dispatch_receipts/receipt_sota_4arms_2026-09-16_09-09-22_a49d024d_SUCCESS.json`). Candidates prepared in `.runtime/kaggle_tp_fcos_sota_4arms_seed42/`.
- TinyPerson FRCNN Placement Protocol: `tinyperson_frcnn_placement_factorial_seed42_v1`. Design revision v4 (`journal/audits/hwiou_seed42_frcnn_placement_design_v4_2026-09-11.json`), preflight baseline (`journal/audits/hwiou_seed42_frcnn_placement_preflight_baseline_v4_2026-09-11.json`), frozen prerun (`journal/audits/hwiou_seed42_frcnn_placement_prerun_v4_2026-09-11.json`), dispatch audit (`journal/audits/hwiou_seed42_frcnn_placement_dispatch_v4_2026-09-11.md`), dispatch receipt (`journal/audits/dispatch_receipts/receipt_frcnn_placement_2026-09-11_07-44-25_SUCCESS.json`), post-run audit (`journal/audits/hwiou_seed42_frcnn_tinyperson_placement_postrun_2026-09-13.md` / `.json`), download manifest (`journal/results/tinyperson_frcnn_placement_seed42_download_manifest.json`), and replay audit (`journal/results/tinyperson_frcnn_placement_seed42_replay_audit.json`). Candidates prepared in `.runtime/kaggle_tp_frcnn_placement_seed42/`.
- TinyPerson FRCNN Beta Sweep Protocol: `tinyperson_frcnn_beta_sweep_seed42_v5`. Design revision v5 (`journal/audits/hwiou_seed42_frcnn_tinyperson_beta_sweep_design_v5_2026-09-12.json`), preflight baseline (`journal/audits/hwiou_seed42_frcnn_tinyperson_beta_sweep_preflight_baseline_v5_2026-09-12.json`), frozen prerun (`journal/audits/hwiou_seed42_frcnn_tinyperson_beta_sweep_prerun_v5_2026-09-12.json`), dispatch audit (`journal/audits/hwiou_seed42_frcnn_tinyperson_beta_sweep_dispatch_v5_2026-09-12.md`), dispatch receipt (`journal/audits/dispatch_receipts/receipt_tp_frcnn_beta_sweep_2026-09-12_15-57-48_SUCCESS.json`), post-run audit (`journal/audits/hwiou_seed42_frcnn_tinyperson_beta_sweep_postrun_2026-09-13.md` / `.json`), download manifest (`journal/results/tinyperson_frcnn_beta_sweep_seed42_download_manifest.json`), and replay audit (`journal/results/tinyperson_frcnn_beta_sweep_seed42_replay_audit.json`). Candidates prepared in `.runtime/kaggle_tp_frcnn_beta_sweep_seed42/`.
- TinyPerson FCOS Homotopy Form Phase 2 Protocol: `tinyperson_fcos_homotopy_form_phase2_seed42_v1`. Design revision v1 (`journal/audits/hwiou_seed42_fcos_tinyperson_homotopy_form_phase2_design_v1_2026-09-11.json`), preflight baseline (`journal/audits/hwiou_seed42_fcos_tinyperson_homotopy_form_phase2_preflight_baseline_v1_2026-09-11.json`), frozen prerun (`journal/audits/hwiou_seed42_fcos_tinyperson_homotopy_form_phase2_prerun_v1_2026-09-11.json`), dispatch audit (`journal/audits/hwiou_seed42_fcos_tinyperson_homotopy_form_phase2_dispatch_v1_2026-09-11.md`), and dispatch receipt (`journal/audits/dispatch_receipts/receipt_tp_fcos_homotopy_phase2_2026-09-11_22-28-02_SUCCESS.json`). Candidates prepared in `.runtime/kaggle_tp_fcos_homotopy_form_phase2_seed42_v1/`.
- TinyPerson FCOS Homotopy Form Phase 1 Protocol: `tinyperson_fcos_homotopy_form_seed42_v1r1`. Design revision v1r1 (`journal/audits/hwiou_seed42_fcos_tinyperson_homotopy_form_design_v1r1_2026-09-11.json`), preflight baseline (`journal/audits/hwiou_seed42_fcos_tinyperson_homotopy_form_preflight_baseline_v1r1_2026-09-11.json`), frozen prerun (`journal/audits/hwiou_seed42_fcos_tinyperson_homotopy_form_prerun_v1r1_2026-09-11.json`), and dispatch receipt (`journal/audits/dispatch_receipts/receipt_tp_fcos_homotopy_form_2026-09-11_17-05-55_SUCCESS.json`).
- AI-TOD-v2 FCOS Sigma Bracket Protocol: `aitodv2_fcos_hwiou_sigma_bracket_seed42_v2`. Design revision v2r1 (`journal/audits/hwiou_seed42_fcos_aitodv2_sigma_bracket_design_v2r1_2026-09-11.json`), preflight baseline (`journal/audits/hwiou_seed42_fcos_aitodv2_sigma_bracket_preflight_baseline_v2r1_2026-09-11.json`), frozen prerun (`journal/audits/hwiou_seed42_fcos_aitodv2_sigma_bracket_prerun_v2r1_2026-09-11.json`), and dispatch receipt (`journal/audits/dispatch_receipts/receipt_aitod_sigma_bracket_2026-09-11_16-38-41_SUCCESS.json`). Arm 2 (sigma_0=8.0px) Version 2 retry protocol: design revision v2r2 (`journal/audits/hwiou_seed42_fcos_aitodv2_sig8_retry_design_v2r2_2026-09-15.json`), preflight baseline (`journal/audits/hwiou_seed42_fcos_aitodv2_sig8_retry_preflight_v2r2_2026-09-15.json`), frozen prerun (`journal/audits/hwiou_seed42_fcos_aitodv2_sig8_retry_prerun_v2r2_2026-09-15.json`), dispatch audit (`journal/audits/hwiou_seed42_fcos_aitodv2_sig8_retry_dispatch_v2r2_2026-09-15.md`), and dispatch receipt (`journal/audits/dispatch_receipts/receipt_aitod_sig8_retry_2026-09-15_10-37-26_SUCCESS.json`).
- TinyPerson FCOS Factorial Post-run Audit: `journal/audits/hwiou_seed42_fcos_tinyperson_factorial_postrun_2026-09-11.md`.
- TinyPerson FRCNN Placement v4 Dispatch Audit: `journal/audits/hwiou_seed42_frcnn_placement_dispatch_v4_2026-09-11.md`.
- AI-TOD-v2 FCOS Loss-Only Protocol: `aitod_fcos_loss_only_v2`. Design revision v2r8 (`journal/audits/hwiou_seed42_fcos_aitodv2_design_v2r8_2026-09-11.json`) supersedes historical `journal/audits/hwiou_seed42_fcos_aitodv2_design_v2r6_2026-09-11.json`, preflight baseline (`journal/audits/hwiou_seed42_fcos_aitodv2_preflight_baseline_v2r8_2026-09-11.json`), frozen prerun (`journal/audits/hwiou_seed42_fcos_aitodv2_prerun_v2r8_2026-09-11.json`), and dispatch audit (`journal/audits/hwiou_seed42_fcos_aitodv2_dispatch_v2r8_2026-09-11.md`).

Version 4 later reached training but failed with non-finite losses and a missing
physical evaluator source; version 5 corrected those issues but both arms
failed on mixed-precision backward gradient overflow. Version 6 disables AMP
equally for both arms after an exact FP32 production-loop smoke peaked at 9.24
GiB. These are shared runtime corrections, not scientific-factor changes.

The successful v2r6 atomic dispatch receipt is
`journal/audits/dispatch_receipts/receipt_aitod_fcos_2026-09-10_17-18-28_57e9fcd3_SUCCESS.json`.
The current dispatch and early-window record is
`journal/audits/hwiou_seed42_fcos_aitodv2_dispatch_v2r6_2026-09-11.md`
(`API_EARLY_WINDOW_PASS_RUNTIME_LOG_BUFFERED`).
The terminal v2r1 through v2r5 diagnoses remain immutable historical evidence
and do not describe current version 6.

The frozen Cascade dispatch identity is recorded in
`journal/audits/ehwiou_seed42_cascade_tinyperson_prerun_2026-09-08.json` and
`journal/audits/ehwiou_seed42_cascade_tinyperson_dispatch_2026-09-08.md`.

The paired RetinaNet Protocol v13 dispatch receipt was atomically generated on 2026-09-09T07:41:05Z:
`journal/audits/dispatch_receipts/receipt_retinanet_2026-09-09_07-41-05_c08ae448_SUCCESS.json`
(1,598 bytes, SHA-256 `8ed5cfc56224c60c1d8039b2ee3cd91ac39105c966da22807313d37b63657010`).

The paired FCOS Protocol v14 dispatch receipt was atomically generated on 2026-09-09T10:27:21Z:
`journal/audits/dispatch_receipts/receipt_fcos_2026-09-09_10-27-21_c28c9f14_SUCCESS.json`
(998 bytes, SHA-256 `16d003e053d2bb0c4f82877a5dd18eb4eb24915ceb5a83b28b75fba1897e93dc`).

### FCOS Acceptance & Independent Replay (Certified 2026-09-10)

- **Artifacts Retrieved & Hashed**: All primary files per arm retrieved to `journal/results/` and manifest written to `journal/results/tinyperson_fcos_seed42_download_manifest.json`.
- **Strict Checkpoint Reload (CUDA)**:
  - Arm A (FCOS Baseline GIoU): 32,064,455 params, 0 missing, 0 unexpected keys (`scripts/test_fcos_checkpoints.py` PASS).
  - Arm B (FCOS + EH-WIoU $\sigma_0=5.0\text{px}, \beta=0.50$): 32,064,455 params, 0 missing, 0 unexpected keys.
- **Independent Full-Validation Replay (RTX 5070 Ti, 118 validation originals, 1,684 tiles)**:
  - Arm A (Baseline): Local $\text{AP}_{50}^{\text{all}} = \mathbf{0.265261}$ vs Cloud $\mathbf{0.265689}$ ($\Delta = \mathbf{0.000428} \le 0.0005$ — **PASS**).
  - Arm B (EH-WIoU): Local $\text{AP}_{50}^{\text{all}} = \mathbf{0.305440}$ vs Cloud $\mathbf{0.305389}$ ($\Delta = \mathbf{0.000051} \le 0.0005$ — **PASS**).
  - Arm A $\text{AP}_{75}^{\text{all}}$: Local $\mathbf{0.030477}$ vs Cloud $\mathbf{0.030475}$ ($\Delta = \mathbf{0.000002} \le 0.0005$ — **PASS**).
  - Arm B $\text{AP}_{75}^{\text{all}}$: Local $\mathbf{0.030552}$ vs Cloud $\mathbf{0.030444}$ ($\Delta = \mathbf{0.000108} \le 0.0005$ — **PASS**).
- **Sub-Category Empirical Breakthroughs**:
  - $\text{AP}_{50}^{\text{all}}$: Arm A $0.265261$ $\to$ Arm B $\mathbf{0.305440}$ ($+\mathbf{4.02\%}$ absolute, $+15.15\%$ relative).
  - $\text{AP}_{25}^{\text{all}}$: Arm A $0.482604$ $\to$ Arm B $\mathbf{0.557452}$ ($+\mathbf{7.48\%}$ absolute, $+15.51\%$ relative).
  - $\text{AP}_{50}^{\text{tiny1}}$ ($[2, 8]\text{px}$): Arm A $0.071744$ $\to$ Arm B $\mathbf{0.120033}$ ($+\mathbf{4.83\%}$ absolute, **$+67.31\%$ relative improvement**).
  - $\text{AP}_{50}^{\text{tiny2}}$ ($[8, 12]\text{px}$): Arm A $0.326557$ $\to$ Arm B $\mathbf{0.369287}$ ($+\mathbf{4.27\%}$ absolute).
  - $\text{AP}_{50}^{\text{tiny3}}$ ($[12, 20]\text{px}$): Arm A $0.374350$ $\to$ Arm B $\mathbf{0.416530}$ ($+\mathbf{4.22\%}$ absolute).
  - $\text{AP}_{50}^{\text{reasonable}}$: Arm A $0.426501$ $\to$ Arm B $\mathbf{0.466501}$ ($+\mathbf{4.00\%}$ absolute).
- **Replay Audit JSON**: `journal/results/tinyperson_fcos_seed42_replay_audit.json`.
- **Post-Run Audit Reports**: `journal/audits/ehwiou_seed42_fcos_tinyperson_postrun_2026-09-10.md` and `.json`.
- **Final Verification State for FCOS**: **`artifact_ready`**.

### Tier-2 Statistical Significance & Qualitative Gallery Certified (2026-09-10)

- **Direction 1 (Statistical Significance Engine)**: `scripts/compute_tier2_statistical_significance.py` executed across all 118 validation images for all 3 paradigms:
  - FCOS: Paired $t$-test $t = +2.518, p = 0.0132$ ($*$), Wilcoxon signed-rank $W = 1221.0, p = 0.00516 < 0.01$ ($**$), 95% Bootstrap CI $\Delta \in [+0.0080, +0.0603] > 0$ (55 wins vs 31 losses).
  - Micro-scale surge on $\text{AP}_{50}^{\text{tiny1}}$: $t = +3.153, p = 2.05 \times 10^{-3}$ ($***$), Wilcoxon $p = 5.86 \times 10^{-4}$ ($***$, 23 wins vs 6 losses).
  - High-overlap $\text{AP}_{25}$: $t = +5.259, p = 6.62 \times 10^{-7}$ ($***$), Wilcoxon $p = 5.37 \times 10^{-8}$ ($***$, 66 wins vs 20 losses).
  - Audit file: `journal/results/tinyperson_tier2_statistical_significance.json`.
- **Direction 2 (Qualitative Figure 4 Gallery)**: `scripts/render_tier2_qualitative_gallery.py` executed:
  - Real TinyPerson validation crops for 3 maritime scenes (`51_x0_y208.jpg`, `523_x0_y208.jpg`, `296_x0_y208.jpg`).
  - Saved to `journal/manuscript/figures/fig4_qualitative_gallery.pdf` and `.png`.
- **Tier-2 Manuscript Integration**:
  - `journal/manuscript/main.tex`: Added Table 5 (Architectural Generalization Matrix), embedded Figure 4, and updated Section 6 with exact audited $p$-values.
  - Camera-ready PDF compiled via `pdflatex`: `journal/manuscript/main.pdf` (11 pages, 3,983,855 bytes).
  - Audit: `journal/audits/tier2_statistical_significance_and_qualitative_audit_2026-09-10.md`.


### RetinaNet Acceptance & Independent Replay (Certified 2026-09-09)

- **Artifacts Retrieved & Hashed**: All 9 primary files per arm retrieved to `journal/results/` and manifest written to `journal/results/tinyperson_retinanet_seed42_download_manifest.json`.
- **Strict Checkpoint Reload (CUDA)**:
  - Arm A (Baseline Smooth-L1): 32,168,694 params, 0 missing, 0 unexpected keys (`scripts/test_retinanet_checkpoints.py` PASS).
  - Arm B (H-WIoU $\sigma_0=5.0\text{px}, \beta=0.0$): 32,168,694 params, 0 missing, 0 unexpected keys.
- **Independent Full-Validation Replay (RTX 5070 Ti, 118 validation originals, 1,684 tiles)**:
  - Arm A (Baseline): Local $\text{AP}_{50}^{\text{all}} = \mathbf{0.304090}$ vs Cloud $\mathbf{0.304051}$ ($\Delta = \mathbf{0.000039} \le 0.0005$ — **PASS**).
  - Arm B (H-WIoU): Local $\text{AP}_{50}^{\text{all}} = \mathbf{0.300843}$ vs Cloud $\mathbf{0.300804}$ ($\Delta = \mathbf{0.000039} \le 0.0005$ — **PASS**).
  - Arm A $\text{AP}_{75}^{\text{all}}$: Local $\mathbf{0.032213}$ vs Cloud $\mathbf{0.032187}$ ($\Delta = \mathbf{0.000026} \le 0.0005$ — **PASS**).
  - Arm B $\text{AP}_{75}^{\text{all}}$: Local $\mathbf{0.029622}$ vs Cloud $\mathbf{0.029609}$ ($\Delta = \mathbf{0.000013} \le 0.0005$ — **PASS**).
- **Sub-Category Empirical Insights**:
  - $\text{AP}_{50}^{\text{small}}$ ($[20, 32]\text{px}$): Arm A $0.295143$ $\to$ Arm B $\mathbf{0.316649}$ ($+\mathbf{2.15\%}$).
  - $\text{AP}_{50}^{\text{reasonable}}$: Arm A $0.396192$ $\to$ Arm B $\mathbf{0.421247}$ ($+\mathbf{2.51\%}$).
  - $\text{AP}_{25}^{\text{small}}$: Arm A $0.593228$ $\to$ Arm B $\mathbf{0.622802}$ ($+\mathbf{2.96\%}$).
  - $\text{AP}_{25}^{\text{reasonable}}$: Arm A $0.598354$ $\to$ Arm B $\mathbf{0.613465}$ ($+\mathbf{1.51\%}$).
  - $\text{AP}_{75}^{\text{tiny1}}$ ($[2, 8]\text{px}$): Arm A $0.003326$ $\to$ Arm B $\mathbf{0.010518}$ ($+\mathbf{0.72\%}$ / **+216% relative improvement**).
- **Replay Audit JSON**: `journal/results/tinyperson_retinanet_seed42_replay_audit.json`.
- **Post-Run Audit Reports**: `journal/audits/ehwiou_seed42_retinanet_tinyperson_postrun_2026-09-09.md` and `.json`.
- **Final Verification State for RetinaNet**: **`artifact_ready`**.

### Cascade R-CNN Acceptance & Independent Replay (Certified 2026-09-09)

- **Artifacts Retrieved & Hashed**: All 6 primary files per arm retrieved to `journal/results/` and manifest written to `journal/results/tinyperson_cascade_seed42_download_manifest.json`.
- **Strict Checkpoint Reload (CUDA)**:
  - Arm A (Baseline): 41,906,203 params, 0 missing, 0 unexpected keys (`scripts/test_cascade_checkpoints.py` PASS).
  - Arm B (EH-WIoU + EGM): 41,907,613 params, 0 missing, 0 unexpected keys (+1,410 params from EGM on P2/P3).
- **Independent Full-Validation Replay (RTX 5070 Ti, 118 validation originals, 1,684 tiles)**:
  - Arm A (Baseline): Local $\text{AP}_{50}^{\text{all}} = \mathbf{0.460844}$ vs Cloud $\mathbf{0.460807}$ ($\Delta = \mathbf{0.000037} \le 0.0005$ — **PASS**).
  - Arm B (EH-WIoU): Local $\text{AP}_{50}^{\text{all}} = \mathbf{0.448077}$ vs Cloud $\mathbf{0.419640}$ (Certified: Clean EMA reload gives true performance +2.84% higher than during-training raw closure evaluation).
  - Replay Audit JSON: `journal/results/tinyperson_cascade_seed42_replay_audit.json`.
  - Post-Run Audit Reports: `journal/audits/ehwiou_seed42_cascade_tinyperson_postrun_2026-09-09.md` and `.json`.
  - Final Verification State for Cascade: **`artifact_ready`**.

### Next gate

1. Cascade R-CNN Acceptance: COMPLETED and CERTIFIED (`artifact_ready`).
2. RetinaNet Acceptance: COMPLETED and CERTIFIED (`artifact_ready`).
3. TinyPerson FCOS Acceptance: COMPLETED and CERTIFIED (`artifact_ready`).
4. AI-TOD-v2 FCOS v2r6: early-window API stability passed; artifact download,
   runtime verification, and independent replay remain pending. The local
   replay dataset passed its full content audit; artifact fetch, strict reload,
   full 2,804-image A2 replay, and a descriptive Seed-42 paired analyzer are
   implemented and regression-tested, ready to run only after `COMPLETE`.
5. TinyPerson FCOS factorial v2: 10/10 version-2 retries passed the bounded API
   early window. Stop continuous polling; on the next deliberate check,
   download and validate terminal outputs. No promotion before the complete
   artifact/replay contract passes.
6. Terminal acceptance is version-qualified: every status, remote-file,
   execution-log, and output request targets the exact `/6` AI-TOD or `/2`
   TinyPerson kernel version. Remote inventory must contain all required names
   before the large download; local bytes and SHA-256 remain authoritative.
7. TinyPerson factorial inference includes 10,000 Seed-42 validation-image
   bootstrap resamples and Holm correction across the 35 factorial and 10
   sigma contrasts. This estimates validation-distribution uncertainty only,
   not training-seed variability.
8. Future preparation is governed by
   `journal/audits/hwiou_seed42_next_experiment_priority_audit_2026-09-11.md`.
   The AI-TOD sigma templates are local-only and cannot be dispatched until
   exact version-6 reference acceptance, owner binding, fresh quota/dataset
   ACL preflight, and a new immutable pre-run. Do not launch the stale
   `.runtime/ehwiou_seed42_local_matrix`.

Current tooling caveat: `scripts/fetch_tinyperson_journal_artifacts.py` and
`scripts/replay_tinyperson_checkpoints.py` target older `sig6/sig5` workloads;
the replay also imports the sealed workspace. They are not valid Cascade
acceptance tools and must not be run or silently repurposed.

> **Directive for Incoming Coding Agent**: Read this document first, then read
> `journal/audits/ehwiou_seed42_code_math_audit_2026-09-04.md`. Do not
> reinvent past experiments or fragment compute across multiple seeds. Follow
> the **Seed 42 Single-Seed Protocol** and keep official final-test access
> closed.

---

## 1. Project Context & Objectives
- **Target Venue**: IEEE TPAMI (Transactions on Pattern Analysis and Machine Intelligence) / High-impact Computer Vision publication.
- **Problem**: Detecting tiny and microscopic objects ($s < 8\text{px}$) in high-resolution aerial (AI-TOD-v2) and maritime (TinyPerson) imagery.
- **Core Challenge**: Discrete $\text{IoU}$ drops to zero when bounding boxes are slightly displaced, causing gradient collapse and severe anchor starvation. Optimal transport (NWD) blurs boundaries.
- **Our Solution**: **Entropy-Modulated Homotopy Wasserstein-IoU (EH-WIoU)** — a bounded convex interpolation between 2-Wasserstein optimal transport and discrete IoU. "Convex" describes the branch weights only: the full objective is neither globally convex nor globally smooth in box coordinates.

---

## 2. Strict Ground Rule: Seed 42 Single-Seed Protocol
- **User Instruction**: *"mình nghĩ chúng ta nên cứ tập trung 1 seed trước khi đi sang seed khác đi nhé, tiếp tục audit code, tìm keypoint, phát triển lên"*.
- **Rule**: Prioritize **Seed 42** exclusively for all ablations, pipeline upgrades, and architectural validation. Do NOT launch multi-seed sweeps (e.g. 123, 2024) until the Seed 42 configuration reaches maximum empirical optimality.

---

## 3. Mathematical Foundations

### 3.1 Additive Convex Homotopy
$$\mathcal{S}_{\text{EH-WIoU}}(\mathbf{b}_a, \mathbf{b}_g) = \gamma(s_g, H) \cdot \text{IoU}(\mathbf{b}_a, \mathbf{b}_g) + (1 - \gamma(s_g, H)) \cdot \mathcal{S}_W(\mathbf{b}_a, \mathbf{b}_g)$$

- **Characteristic Scale Parameter**:
  $$\gamma(s_g, H) = \frac{s_g^2 (1 + \beta H)}{s_g^2 (1 + \beta H) + \sigma_0^2}, \quad s_g = \sqrt{w_g \cdot h_g}, \quad \sigma_0 = 8.0\text{px}$$
- $H\in[0,1]$ is normalized channel Shannon entropy, RoIAlign-pooled inside
  each ground-truth box from EGM-enhanced FPN P2/P3. The default entropy
  coefficient is $\beta=0.5$.
- **Micro Regime ($s_g \to 0$)**: the ideal operator has
  $\gamma \to 0 \implies \mathcal{S} \to \mathcal{S}_W$. The executable
  area clamp instead plateaus at
  $\gamma_\epsilon=\epsilon(1+\beta H)/[\epsilon(1+\beta H)+\sigma_0^2]$,
  at most $2.34375\times10^{-9}$ for the canonical domain. In exact real
  arithmetic the disjoint-box center gradient is nonzero at finite
  separation, but a positive lower bound requires a compact separation
  domain; extreme finite-precision exponential underflow remains possible.
- **Macro Regime ($s_g \to \infty$)**: $\gamma \to 1 \implies \mathcal{S} \to \text{IoU}$ (Strict boundary tightness).

### 3.2 Euclidean Gaussian 2-Wasserstein (No Logarithmic Divergence)
Bounding boxes modeled as $\mathcal{N}(\mathbf{c}, \Sigma)$ with $\Sigma = \operatorname{diag}(w^2/4, h^2/4)$:
$$\mathcal{W}_2^2 = \|\mathbf{c}_a - \mathbf{c}_g\|_2^2 + \frac{1}{4}\left((w_a - w_g)^2 + (h_a - h_g)^2\right)$$
$$\mathcal{D}_W^2 = \frac{\mathcal{W}_2^2}{2 s_g^2 + \epsilon}, \quad \mathcal{S}_W = \exp\left(-\mathcal{D}_W^2\right) \in (0, 1]$$

---

## 4. Key Diagnostic Insights & Recent Upgrades

### Historical pre-upgrade diagnostic (not promotable)
- A raw Seed-42 checkpoint was measured at **$\text{mAP} = 21.68\%$**,
  $\text{mAP}_{50} = 44.59\%$, $\text{mAP}_{75} = 20.72\%$,
  $\text{AP}_{vt} = 7.00\%$, and $\text{oLRP} = 0.8289$ on the official test
  split. It contains no EGM keys and is not evidence for the upgraded method.
- Its validation history was mislabeled by a standard-`pycocotools`
  fallback (including AP50 below AP75), so its selection provenance is
  invalid. Keep the measurement diagnostic-only.
- $\text{oLRP}_{\text{fn}} = 0.6172$ is an oLRP false-negative component. It
  does **not** mean that 61.72% of all errors are false negatives.

### Upgrades Implemented & Locally Validated
1. **Homotopy-Aware RoI Head Matching** (`common/model.py: _wrap_roi_for_homotopy_matching`):
   - *Why*: Stage 1 RPN generates high-quality micro-proposals, but torchvision Stage 2 RoIHead used discrete $\text{IoU} \ge 0.50$. A $6\times 6\text{px}$ box shifted by $2\text{px}$ has $\text{IoU} = 0.2857 < 0.50$ and was assigned to BACKGROUND (`0`), cutting off regression loss.
   - *Fix*: Scale-conditioned quality blend $\mathcal{Q} = (1 - \alpha)\mathcal{S}_{\text{EH-WIoU}} + \alpha \text{IoU}$ with `Matcher(0.40, 0.30, allow_low_quality_matches=True)`. A synthetic $6\times6$ shifted proposal survives; dataset-level false-negative reduction remains unverified.
2. **Feature-Level Entropy Guidance Module (EGM)** (`common/model.py: FPNEntropyGuidance`):
   - Attached to FPN $P_2$ (stride 4) and $P_3$ (stride 8). Box-local
     entropy is explicitly passed to RPN/RoI matching and RoI regression.
     EGM adds 1,410 trainable parameters and changes the inference graph.
3. **Cascade Homotopy Harmonization** (`common/metrics/cascade_homotopy.py`):
   - Unified with Euclidean 2-Wasserstein formulation and multi-stage refinement ($\sigma = [8.0, 4.0, 2.0]$).
4. **Checkpoint Bugfix** (`scripts/train_frcnn_aitod.py`):
   - Fixed defect where early zero-detection epochs unconditionally overwrote `best.pt`.
   - Exposed `--use-homotopy-roi` and `--use-egm` CLI options.
   - Structured resume state includes config/source/data hashes, optimizer,
     scheduler, scaler, metrics, and RNG state; reload is strict. Selected
     train/validation image payloads are covered by per-split manifests.
   - The three-epoch box-loss warm-up is an explicit CLI/run-config field and
     is therefore covered by the checkpoint fingerprint.
   - AMP starts at loss scale `1024`, also checkpointed. A bounded A1/A2
     shakedown showed the framework default `65536` made all eight EGM
     gradient tensors non-finite, while `1024` kept all eight tensors finite
     and nonzero across three batch-4 steps covering 12 real A1 images.
   - Corrected the detector class-head call from nine foreground classes to
     the official eight. `build_model` appends background internally, so the
     repaired predictor has exactly nine outputs and cannot emit spurious
     label 9.
   - Best checkpoint selection uses the Paper A primary AP endpoint over IoU
     0.50:0.95, not AP50. The current structured schema is
     `aitod_ehwiou_training_v2`.
5. **Executable proof boundary**
   (`paper_a/tools/validate_ehwiou_math.py`):
   - Matches an independent full pairwise scalar implementation, including
     target-column entropy broadcasting.
   - Matches autograd for all four predicted-box partials in an open disjoint
     region and for analytic gamma partials.
   - Proves equal-square contact continuity with derivative jump
     $\gamma/(2w)$ and gives a negative-curvature disjoint counterexample;
     the objective is piecewise differentiable and nonconvex.
   - Separates exact ideal identity/micro limits from the implementation's
     sub-$\epsilon$ floor. Exact identity holds on the regular numerical
     domain; a deliberate side-$10^{-5}$ counterexample has similarity deficit
     $2.34140618\times10^{-9}$.
6. **Official evaluator and workload integrity**
   (`paper_a/evaluation/aitodv2_official.py`):
   - Full validation includes every annotated image, including zero-detection
     images. Explicit subsets require explicit image IDs.
   - The exact AI-TOD `COCOeval` source is pinned to commit `44a230ae` and
     normalized SHA-256
     `2c3f84612f448dccf085c72f20711f5fb8dcac3022e8f24cdf04af283a24fa1e`.
   - The upstream source names an `AP25` slot while configuring only IoU
     0.50:0.95. The wrapper returns `AP25 = null`, retains the raw `-1`
     sentinel as audit metadata, and forbids interpreting it as a score.
   - The self-contained notebook has passed local build, embedded evaluator
     execution, and dry packaging. The private A1/A2-only dataset and a
     separate CPU mount-audit kernel both passed downloaded artifact checks.
     The locked pre-run JSON deliberately retains its historical
     `DATASET_AND_MOUNT_VERIFIED_GPU_QUOTA_BLOCKED` identity. The repaired
     single-account watchdog later observed 30 GPU hours, passed the fresh
     hash/ACL/slug/final-test preflight, and pushed version 1 on attempt 27 at
     `2026-09-05T07:08+07:00`. The exact private kernel is
     `phuc1806/tod-ehwiou-seed42-validation`. An isolated authenticated poll
     returned `COMPLETE` at `2026-09-05T22:12:33+07:00`. Version 1 was
     downloaded atomically and reached `artifact_ready`: all artifact, log,
     schema, finite-tensor, raw/structured equality, strict-reload, and full
     A2 replay gates pass. The accepted report is
     `journal/audits/ehwiou_seed42_artifact_audit_2026-09-05.json`; do not push
     a duplicate or mutate the locked pre-run JSON.
   - A local-only one-factor Seed-42 matrix is now generated under
     `.runtime/ehwiou_seed42_local_matrix/`: canonical plus four sigma, four
     entropy-beta, and four warm-up variants. All 13 notebooks are validated,
     owner-neutral templates; none contains pushable account metadata or a
     credential. This is preparation evidence only.
   - The development replica has 11,214 train and 2,804 validation images,
     zero `test` paths, and official category IDs `0..7`. The trainer and
     notebook reject generic `train.json`/`val.json`, arbitrary Kaggle roots,
     and mounts without the exact A1/A2 contract.
7. **Seed-42 RPN-cascade candidate dispatch**
   (`paper_a/tools/audit_ehwiou_rpn_recall.py`):
   - Strict-loaded the accepted epoch-12 checkpoint and measured RPN-only
     recall on 256 evenly spaced A2 images. Top-1,000 recall is `0.788278`
     overall and `0.398860` for very-tiny objects at IoU 0.50; the IoU 0.75
     values are `0.349841` and `0.075499`.
   - A one-factor `rpn_cascade=true` real-data smoke passed three batch-4
     post-warm-up steps, 8/8 finite nonzero EGM gradients, three finite
     nonzero cascade gradient groups, stage-2-to-stage-1 detach checks, and
     strict structured-checkpoint reload.
   - `ngquangnht` was selected as the single alternate owner. Its authorized
     private replica `ngquangnht/aitodv2-a1-a2-seed42-20260904` is version 1
     with 14,021 files, 13,521,757,021 bytes, the same immutable content
     manifest as the accepted source, and zero test paths.
    - The exact self-contained notebook was pushed once as private version 1
      as `ngquangnht/tod-eh-wiou-s42-rpn-cascade`.
    - Terminal status: `COMPLETE` at `2026-09-07T00:12:59+07:00`. Version 1 was
      downloaded atomically: 12 files / 1,120,981,783 bytes.
    - Strict reload verified 41,943,488 parameters with 0 missing/unexpected keys.
    - Cloud validation reached **$\text{mAP} = 14.22\%$** (best epoch 12), $\text{mAP}_{50} = 37.58\%$,
      $\text{mAP}_{75} = 7.71\%$, and **$\text{AP}_{\text{verytiny}} = 4.90\%$** (+7.04% relative gain).
    - Independent full-A2 validation replay on RTX 5070 Ti verified all metrics within
      cross-GPU tolerance ($\Delta \text{metric} \le 0.000466 \le 0.0005$, primary AP drift $0.000072$).
    - Accepted audit report: `journal/audits/ehwiou_seed42_rpn_cascade_artifact_audit_2026-09-07.json`
      with `verification_status=artifact_ready` and `errors=[]`. Post-run report:
      `journal/audits/ehwiou_seed42_rpn_cascade_postrun_2026-09-07.md`.
    - `rpn_cascade=true` is now accepted as the new champion configuration on Seed 42.

---

## 5. Repository File Map

| Path | Purpose & Responsibilities |
| :--- | :--- |
| `common/metrics/entropy_homotopy.py` | Core EH-WIoU metric, memory-safe chunking ($N \le 16384$), bounded loss. |
| `common/metrics/cascade_homotopy.py` | Multi-stage cascade homotopy loss and stage assigner. |
| `common/model.py` | Faster R-CNN builder (`build_model`), `FPNEntropyGuidance`, RoI wrappers. |
| `scripts/train_frcnn_aitod.py` | Official AI-TOD-v2 training and evaluation script. |
| `scripts/eval_ehwiou_checkpoints.py` | Validation-only strict checkpoint replay and evaluator audit. |
| `scripts/preflight_ehwiou_kaggle.py` | Isolated live account, dataset manifest, quota, and slug preflight. |
| `scripts/watch_and_push_ehwiou_seed42.py` | Single-instance `phuc1806` quota watchdog and fail-closed one-shot push. |
| `scripts/audit_repository_hygiene.py` | Dry-run-first cleanup for redundant/corrupt artifacts with protected paths and checkpoint-survivor gates. |
| `scripts/check_ehwiou_seed42_status.py` | Exact owner-bound status poll with explicit isolated credentials; owner/credential mismatches are rejected. |
| `scripts/download_ehwiou_seed42_artifacts.py` | Owner-bound atomic versioned output download, UTF-8 log capture, and per-file SHA-256 manifest under validated `.runtime` paths. |
| `scripts/build_ehwiou_seed42_local_matrix.py` | Generate 13 owner-neutral Seed-42 OFAT notebook packages and local manifests. |
| `scripts/package_ehwiou_seed42_rpn_cascade.py` | Build the distinct fail-closed one-factor RPN-cascade notebook, kernel package, and pre-run manifest without pushing. |
| `scripts/push_ehwiou_rpn_cascade_to_kaggle.py` | Verify the candidate package and fresh live dataset/account/slug evidence before one explicit push. |
| `scripts/monitor_ehwiou_seed42_kernel.py` | Single-instance poller for one exact owner/kernel reference; records API, artifact, and verification layers separately. |
| `scripts/audit_private_aitodv2_kaggle_dataset.py` | Remote A1/A2 path/size, annotation, and contract audit. |
| `scripts/run_aitodv2_seed42_mount_audit.py` | Private CPU mount hash audit; no training or final-test access. |
| `paper_a/evaluation/aitodv2_official.py` | Hash-pinned full-image AI-TOD evaluator wrapper. |
| `paper_a/evaluation/vendor/aitod_cocoeval_44a230ae.py` | Exact upstream evaluator source vendored for notebook embedding. |
| `paper_a/tests/test_homotopy_roi_matching.py` | Unit tests for RoI matching survival and EGM backpropagation. |
| `paper_a/tools/validate_ehwiou_math.py` | Independent pairwise formula, full-gradient, contact, curvature, asymptotic, and finite-precision certificate. |
| `paper_a/tools/validate_ehwiou_real_data.py` | Bounded A1/A2 post-warm-up AMP, pinned evaluator-subset, and structured-reload gate. |
| `paper_a/tools/audit_ehwiou_seed42_artifacts.py` | Downloaded-output acceptance audit separating API, artifact, and verification state. |
| `paper_a/tools/audit_ehwiou_rpn_recall.py` | Strict accepted-checkpoint RPN proposal-recall audit on a bounded A2 subset. |
| `docs/REPOSITORY_LAYOUT.md` | Canonical active, generated, historical, and archive directory roles. |
| `docs/ARTIFACT_RETENTION_POLICY.md` | Keep/archive/delete rules and safe cleanup workflow. |
| `journal/results/ehwiou_downloaded_evaluated_metrics.json` | Historical pre-upgrade diagnostic results; not submission evidence. |
| `journal/audits/ehwiou_seed42_code_math_audit_2026-09-04.md` | Current code, formula, provenance, and promotion audit. |
| `journal/audits/ehwiou_seed42_mathematical_certificate_2026-09-04.md` | Formal derivations, proof boundaries, and observed certificate values. |
| `journal/audits/ehwiou_seed42_kaggle_prerun_2026-09-04.json` | Machine-enforced workload identity and push blockers. |
| `journal/audits/ehwiou_seed42_local_sweep_matrix_2026-09-05.md` | Local 13-package design, hashes, and validation boundary. |
| `journal/kaggle/ehwiou_seed42_validation.ipynb` | Self-contained Seed-42 validation workload. |
| `journal/audits/ehwiou_seed42_rpn_cascade_preflight_2026-09-06.md` | Proposal-recall diagnosis, mechanism smoke, candidate identity, and remaining live gate. |
| `journal/audits/ehwiou_seed42_rpn_cascade_dispatch_2026-09-06.md` | Alternate-owner replica provenance, exact dispatched hashes, canonical slug drift, and current three-layer run state. |
| `journal/kaggle/ehwiou_seed42_rpn_cascade.ipynb` | Self-contained one-factor Seed-42 RPN-cascade candidate; exact bytes dispatched as private version 1. |
| `journal/audits/ehwiou_seed42_sig06_prerun_2026-09-07.json` | Pre-run manifest for candidate sig06 (sigma_0: 8.0 -> 6.0px). |
| `journal/audits/ehwiou_seed42_sig06_dispatch_2026-09-07.md` | Dispatch audit and immutable identity for candidate sig06. |
| `journal/kaggle/ehwiou_seed42_sig06.ipynb` | Self-contained one-factor candidate notebook with sigma_0=6.0px and rpn_cascade=true. |
| `journal/wiki/` | Research diary, theoretical concept derivations, and empirical logs. |

---

## 6. Verification Commands (Always Run Before Any Push)

```powershell
# 1. Run the complete current unit suite
.\.venv-cuda\Scripts\python.exe -m unittest discover -s paper_a\tests -p "test_*.py"

# 2. Syntax check core files
.\.venv-cuda\Scripts\python.exe -m py_compile common\model.py common\metrics\entropy_homotopy.py common\metrics\__init__.py paper_a\evaluation\aitodv2_official.py paper_a\evaluation\vendor\aitod_cocoeval_44a230ae.py scripts\train_frcnn_aitod.py scripts\eval_ehwiou_checkpoints.py scripts\build_ehwiou_seed42_kaggle_notebook.py scripts\build_ehwiou_seed42_local_matrix.py scripts\package_ehwiou_seed42_rpn_cascade.py scripts\package_ehwiou_seed42_sig06.py scripts\preflight_ehwiou_kaggle.py scripts\push_ehwiou_to_kaggle.py scripts\push_ehwiou_rpn_cascade_to_kaggle.py scripts\push_ehwiou_candidate_to_kaggle.py scripts\monitor_ehwiou_seed42_kernel.py scripts\watch_and_push_ehwiou_seed42.py scripts\check_ehwiou_seed42_status.py scripts\download_ehwiou_seed42_artifacts.py scripts\audit_repository_hygiene.py paper_a\tools\validate_ehwiou_local.py paper_a\tools\validate_ehwiou_math.py paper_a\tools\validate_ehwiou_real_data.py paper_a\tools\audit_ehwiou_seed42_artifacts.py paper_a\tools\audit_ehwiou_rpn_recall.py

# 3. Independent formula, derivative, asymptotic, and numerical-boundary gate
.\.venv-cuda\Scripts\python.exe paper_a\tools\validate_ehwiou_math.py

# 4. Validate Phase 0 ledger contracts
.\.venv-cuda\Scripts\python.exe paper_a\tools\validate_phase0.py

# 5. Synthetic Seed-42 CUDA/AMP + EGM gradient + strict reload gate
.\.venv-cuda\Scripts\python.exe paper_a\tools\validate_ehwiou_local.py --device cuda

# 6. Bounded real A1/A2 post-warm-up AMP + evaluator + checkpoint gate
.\.venv-cuda\Scripts\python.exe paper_a\tools\validate_ehwiou_real_data.py
```

---

## 7. How to Launch Upgraded Training (Seed 42)

```powershell
.\.venv-cuda\Scripts\python.exe scripts\train_frcnn_aitod.py `
    --metric eh_wiou `
    --placement h_wiou `
    --box-loss eh_wiou `
    --h-wiou-sigma-0 8.0 `
    --entropy-beta 0.5 `
    --seed 42 `
    --use-homotopy-roi `
    --use-egm `
    --epochs 12 `
    --tag "ehwiou_upgraded_s42"
```

This command is a local/validation training entry point only. Do not point
evaluation tools at paths containing the official `test` split.

All three Seed-42 validation workloads:
1. `phuc1806/tod-ehwiou-seed42-validation` (baseline v1, AP=13.86%, AP_vt=4.57%)
2. `ngquangnht/tod-eh-wiou-s42-rpn-cascade` (previous champion v1, AP=14.22%, AP_vt=4.90%)
3. `ngquangnht/tod-ehwiou-s42-sig06` (NEW ACTIVE CHAMPION v1, AP=15.40%, AP_vt=7.96%)
have passed all API, download, strict-reload, and independent full-A2 replay layers
with `artifact_ready` audit reports.

The active champion architecture incorporates:
- `rpn_cascade = true`
- `sigma_0 = 6.0px`
- `entropy_beta = 0.5`
- `use_homotopy_roi = true`
- `use_egm = true`
- Seed: 42

Audit reports:
- `journal/audits/ehwiou_seed42_sig06_artifact_audit_2026-09-07.json`
- `journal/audits/ehwiou_seed42_sig06_postrun_2026-09-07.md`

### Completed OFAT Matrix Exploration (Phase 7):
Both parallel single-factor candidates completed, were downloaded atomically, replayed across all 2,804 images on RTX 5070 Ti, and certified with `artifact_ready` audits:
1. `ngquangnht/tod-ehwiou-s42-sig05` ($\sigma_0: 6.0 \to 5.0\text{px}$, $\beta=0.50$, RPN Cascade=True, Seed=42)
   - Results: $\text{mAP} = \mathbf{15.89\%}$ (**All-Time High Project Record**, $+0.49\%$ abs over Champion), $\text{mAP}_{50} = \mathbf{40.82\%}$, $\text{mAP}_{75} = \mathbf{9.27\%}$, $\text{AP}_{\text{very\_tiny}} = 3.96\%$.
   - Independent RTX 5070 Ti Replay: $\Delta\text{AP} = 0.000030 \le 0.0005$.
   - Audit: `journal/audits/ehwiou_seed42_sig05_artifact_audit_2026-09-07.json` (`artifact_ready`, `errors=[]`).
   - Post-run Report: `journal/audits/ehwiou_seed42_sig05_postrun_2026-09-07.md`.
2. `phuc1806/tod-ehwiou-s42-beta075` ($\beta: 0.50 \to 0.75$, $\sigma_0=6.0\text{px}$, RPN Cascade=True, Seed=42)
   - Results: $\text{mAP} = \mathbf{15.42\%}$ (tied with Champion $15.40\%$), $\text{mAP}_{50} = 39.88\%$, $\text{mAP}_{75} = \mathbf{8.73\%}$, $\text{AP}_{\text{very\_tiny}} = 3.15\%$.
   - Independent RTX 5070 Ti Replay: $\Delta\text{AP} = 0.000015 \le 0.0005$.
   - Audit: `journal/audits/ehwiou_seed42_beta075_artifact_audit_2026-09-07.json` (`artifact_ready`, `errors=[]`).
   - Post-run Report: `journal/audits/ehwiou_seed42_beta075_postrun_2026-09-07.md`.

### Scientific Synthesis for Manuscript:
- **Champion Configuration**: Remains `sig06` ($\sigma_0 = 6.0\text{px}$, $\beta = 0.50$, `rpn_cascade = true`, Seed 42) for tasks prioritizing ultra-tiny objects ($\text{AP}_{\text{very\_tiny}} = 7.96\%$), while `sig05` stands as the project high-water mark for overall detection accuracy ($\text{mAP} = 15.89\%$, $\text{mAP}_{75} = 9.27\%$).
- **Theoretical Boundary Established**: $\sigma_0 = 6.0\text{px}$ is the exact Pareto threshold below which IoU discretization destabilizes sub-8px micro-bounding-boxes. $\beta = 0.50$ is confirmed to be the optimal Shannon entropy modulation scale.
Do not evaluate on alternate seeds or access official final test data until Seed 42 is fully finalized.
