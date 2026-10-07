---
title: "Journal Overview: Entropy-Modulated Homotopy Wasserstein-IoU"
type: "overview"
created: "2026-08-23"
updated: "2026-09-27"
sources:
  - "common/metrics/entropy_homotopy.py"
  - "common/model.py"
  - "journal/audits/ehwiou_seed42_code_math_audit_2026-09-04.md"
  - "journal/audits/ehwiou_seed42_mathematical_certificate_2026-09-04.md"
  - "journal/audits/ehwiou_seed42_artifact_audit_2026-09-05.json"
  - "journal/audits/ehwiou_seed42_rpn_cascade_preflight_2026-09-06.md"
  - "journal/audits/ehwiou_seed42_rpn_cascade_dispatch_2026-09-06.md"
  - "journal/audits/official_test_manuscript_reconciliation_20260925.json"
  - "journal/audits/manuscript_submission_readiness_audit_20260927_r2.md"
  - "journal/audits/manuscript_final_evidence_reconciliation_20260927.md"
  - "journal/audits/tinyperson_fcos_best_selector_validation_recovery_20260927.md"
  - "journal/audits/manuscript_figure2_publication_source_alignment_20260927.md"
  - "journal/manuscript/figures/fig2_architecture_publication.svg"
  - "journal/manuscript/figures/fig2_architecture_publication.drawio"
  - "journal/manuscript/figures/generate_official_test_overview.py"
  - "journal/manuscript/figures/fig7_official_test_overview.svg"
tags:
  - "journal"
  - "eh-wiou"
  - "seed-42"
---

# Journal Overview: EH-WIoU

The active journal hypothesis is Entropy-Modulated Homotopy
Wasserstein-IoU (EH-WIoU) for tiny-object Faster R-CNN. All current
optimization is restricted to Seed 42. Further official-test evaluation requires new explicit authorization; completed test results are summarized below.

## Canonical method

For box-local normalized P2/P3 Shannon entropy \(H_g\in[0,1]\),

\[
\gamma(s_g,H_g)=
\frac{s_g^2(1+\beta H_g)}
     {s_g^2(1+\beta H_g)+\sigma_0^2},
\]

\[
S_{\mathrm{EH}}(A,G)=
\gamma\,\operatorname{IoU}(A,G)
+(1-\gamma)\exp[-D_W^2(A,G)].
\]

The Gaussian 2-Wasserstein term is Euclidean in box center, width, and height
and uses no logarithmic box transform. The score is directional because target
scale \(s_g\) controls both normalization and homotopy weight; it is a bounded
similarity/objective, not a symmetric mathematical metric.

See [[entropy_homotopy_wiou]] and
`journal/audits/ehwiou_seed42_mathematical_certificate_2026-09-04.md` for the
implementation-level definition and proof boundaries. “Convex” refers only
to the scalar mixture, not convexity in box parameters.

## Architecture

- EGM enhances FPN P2/P3 and adds 1,410 trainable parameters.
- RoIAlign pools normalized channel entropy inside each ground-truth box.
- The resulting prior modulates both RPN assignment passes, Homotopy RoI
  matching, and aligned RoI regression.
- The RoI matcher uses the scale-conditioned blend
  \(Q=(1-\alpha)S_{\mathrm{EH}}+\alpha\operatorname{IoU}\), with
  \(\alpha=\min(1,s_g/32)\).

## Manuscript Figure 2

The manuscript uses `journal/manuscript/figures/fig2_architecture_publication.svg`,
with `fig2_architecture_publication.drawio` as its editable source. The figure
identifies the per-arm SNGD, target-normalized W2, and exponent-capped YOLO
affinity choices, alongside training placements. The former SNGD-only label was
stale and has been refreshed. See
`journal/audits/manuscript_figure2_publication_source_alignment_20260927.md`
for asset hashes and the PDF render review.

## Evidence state

Local implementation gates pass: closed-form and derivative checks,
full-suite unit tests, CUDA/AMP forward-backward, nonzero gradients in every
EGM parameter tensor, and strict upgraded checkpoint reload.

At the time of the beta replay, the manuscript gate remained `HOLD_SUBMISSION` because
the diagnostic loss-only table was still present. That disposition is superseded by the
current readiness audit below, which records its removal. The paired AI-TOD-v2
Faster R-CNN beta-0 arm passes all nine validation metrics in a replay matching the
cloud-reported PyTorch/TorchVision versions and trainer `torch.no_grad()` context.
Beta-0.5 remains held because AP_t, AR100, and AR1500 exceed the unchanged replay
tolerance; the pair supports no beta-effect claim. See
`journal/audits/aitodv2_frcnn_dual_entropy_prior_r2_replay_environment_adjudication_20260927.md`.

The four external AI-TOD-v2 Faster R-CNN assignment rows are withdrawn from the
manuscript table. Their frozen notebook payload, saved `run_config`, and prior
strict-reload factory do not share one source identity, and serialized detector
fields disagree with the frozen trainer's model-build arguments. The earlier
strict-reload `PASS` does not close that config gate. See
`journal/audits/aitodv2_frcnn_assignment_source_config_adjudication_20260927.md`.

The current manuscript audit is
`journal/audits/manuscript_final_evidence_reconciliation_20260927.md`; verdict
`REVIEW_READY_WITH_LIMITATIONS` means ready for internal scientific review, not
venue compliance or submission approval. `tab:sota_losses` is now a
TinyPerson-only Table 3. Its sparse AI-TOD-v2 block was removed; those diagnostics
remain in dated audits because no complete protocol-matched baseline/proposed
pair passed the row acceptance gates. Eight external TinyPerson rows use
terminal metric replay and best-checkpoint prediction artifacts for schema
checks; IGWD, GCD, and Inner-IoU use their predeclared best-AP50 selector for
both. These mixed selectors do not support rankings. Table 4 withholds all
AI-TOD-v2 assignment values; TinyPerson NWD-assignment remains withheld, while
the ATSS/SimD schema-artifact selector split is stated in the prose before
Table 4. RFLA
uses its replay-passing best-AP50 selector. The manuscript retains six figures
and excludes the held AI-TOD-v2 qualitative figure. No current claim requires
retraining or another detector family. A same-detector SNGD-versus-TW2
comparison is future work only if a later claim needs to attribute an empirical
effect to one exact affinity; nothing is queued. All held-out final-test
surfaces remain closed to new inference.

The current manuscript revision uses short descriptive captions for all 12
tables and six figures. Selector and evidence qualifications, interpretations,
measurement details, theorem scope, and figure legends/results now appear in
nearby prose. The source and 21-page A4 PDF hashes are recorded in the current
manuscript audit above; this editorial pass changed no scores or selectors.

Version 1 of `phuc1806/tod-ehwiou-seed42-validation` is accepted as upgraded
validation-only evidence. Its best/final epoch-12 AP is `0.1386449343`, with
AP50 `0.3714829740`, AP75 `0.0713439441`, AP very tiny `0.0457469118`, and
AR100 `0.2251053130`. API state, downloaded artifact integrity, strict reload,
and independent full-A2 replay all pass. The historical 21.68% AP measurement
still belongs to a pre-upgrade checkpoint with no EGM parameters and invalid
validation-selector labels caused by a standard COCO fallback. It remains
diagnostic history only. Likewise,
\(oLRP_{\mathrm{fn}}=0.6172\) is a component value, not proof that 61.72% of
all errors are false negatives.

The current source-grounded decision record is
journal/audits/ehwiou_seed42_code_math_audit_2026-09-04.md. Chronology is
preserved in [[log]], including older claims that have since been corrected.

## Next gate

The bounded Seed-42 proposal-recall audit is complete. On 256 evenly spaced A2
images, top-1,000 RPN recall is `0.788278` overall and `0.398860` for very-tiny
objects at IoU 0.50; at IoU 0.75 it is `0.349841` and `0.075499`. This creates
diagnostic headroom for exactly one controlled `rpn_cascade=true` candidate.

The candidate passed a three-step real-data CUDA/AMP mechanism smoke,
stage-2-to-stage-1 detach checks, finite nonzero cascade/EGM gradients, strict
reload, and self-contained notebook packaging. A fresh isolated preflight on
`ngquangnht` verified its private version-1 A1/A2 replica, 45 GPU hours, zero
test paths, and an unused candidate slug. Exactly one private version was then
pushed. Kaggle accepted the canonical reference
`ngquangnht/tod-eh-wiou-s42-rpn-cascade`; the requested metadata slug omitted
the separator between `eh` and `wiou`, so the dispatch audit records both
references and forbids a retry.

## Champion evidence state

Version 1 of `ngquangnht/tod-eh-wiou-s42-rpn-cascade` has been downloaded and
formally accepted as the new Seed-42 champion. On the audited AI-TOD-v2 A2 validation set:
- Primary AP reaches `0.142235` (up from `0.138645`, +2.59% relative).
- AP50 reaches `0.375809` (up from `0.371483`).
- AP75 reaches `0.077144` (up from `0.071344`, +8.13% relative).
- AP very tiny reaches `0.048968` (up from `0.045747`, +7.04% relative).
- AR100 reaches `0.225227`.

Full independent validation replay on all 2,804 A2 images (RTX 5070 Ti) passed
all cross-GPU criteria (maximum metric drift `0.0004657 <= 0.0005`, primary AP drift `0.0000722`).
The machine-readable report is `journal/audits/ehwiou_seed42_rpn_cascade_artifact_audit_2026-09-07.json`
with `verification_status=artifact_ready` and `errors=[]`.

## Historical Tier-2 Architectural Generalization State (Scale Match TinyPerson, Seed 42)

The following older run summary records its original protocol and is not the current
manuscript acceptance ledger. In particular, its Cascade AP25/tiny1 method cells
conflict with later replay records and are withheld from the current paper; see
`journal/audits/manuscript_cascade_validation_selector_discrepancy_20260926.md`.

The Tier-2 architectural generalization matrix has been completed across all 3 canonical object detection paradigms on Scale Match TinyPerson (`.runtime/local/program_b/b1_tiled_20260814`, 118 validation images, 1,684 tiles):
1. **Two-Stage Multi-Stage (Cascade R-CNN)**:
   - Baseline (`ngquangnht/tod-tp-cas-base-s42`): $\text{AP}_{50}^{\text{all}} = 0.4608$, $\text{AP}_{75}^{\text{all}} = 0.0700$.
   - EH-WIoU (`amongus1504/tod-tp-cas-ehwiou-s42`): clean EMA replay $\text{AP}_{50}^{\text{all}} = 0.4481$; historical AP25 `0.6276` and tiny1 AP50 `0.2743` are **withdrawn** because they disagree with the later replay records.
   - Historical replay record: `journal/results/tinyperson_cascade_seed42_replay_audit.json`; current discrepancy decision: `journal/audits/manuscript_cascade_validation_selector_discrepancy_20260926.md`.
2. **One-Stage Dense Anchor-Based (RetinaNet)**:
   - Baseline (`quangnhtng/tod-tp-ret-base-s42`): $\text{AP}_{50}^{\text{all}} = 0.3041$, $\text{AP}_{75}^{\text{all}} = 0.0322$.
   - H-WIoU (`hngtrngtn/tod-tp-ret-hwiou-s42`): $\text{AP}_{50}^{\text{all}} = 0.3008$, $\text{AP}_{50}^{\text{small}} = \mathbf{0.3166}$ ($+2.15\%$), $\text{AP}_{50}^{\text{reasonable}} = \mathbf{0.4212}$ ($+2.51\%$), $\text{AP}_{75}^{\text{tiny1}} = \mathbf{0.0105}$ ($+216\%$ rel).
   - Certified replay: `journal/results/tinyperson_retinanet_seed42_replay_audit.json` (`artifact_ready`).
3. **One-Stage Anchor-Free (FCOS)**:
   - Baseline (`qnhat1504/tod-tp-fcos-base-s42`): $\text{AP}_{50}^{\text{all}} = 0.2653$, $\text{AP}_{75}^{\text{all}} = 0.0305$.
   - EH-WIoU (`thyngluthy/tod-tp-fcos-ehwiou-s42`): $\text{AP}_{50}^{\text{all}} = \mathbf{0.3054}$ ($+\mathbf{4.02\%}$ abs, $+15.15\%$ rel), $\text{AP}_{25}^{\text{all}} = \mathbf{0.5575}$ ($+7.48\%$), $\text{AP}_{50}^{\text{tiny1}} = \mathbf{0.1200}$ (**$+67.31\%$ relative surge**).
   - Certified replay: `journal/results/tinyperson_fcos_seed42_replay_audit.json` (`artifact_ready`).

All three detector pairs pass strict checkpoint reload. The FCOS and RetinaNet
arms also satisfy the stated cross-GPU metric-drift threshold. The Cascade
method does not: cloud AP50 `0.419640` versus clean EMA replay AP50 `0.448077`
gives drift `0.028437`, so it is accepted as an artifact with a documented
evaluation-path discrepancy, not as evidence of universal `<= 0.0005`
cross-GPU agreement.

## TinyPerson FCOS Factorial Acceptance (2026-09-11)

All 10 arms of the TinyPerson FCOS Factorial completed with `COMPLETE` and were atomically retrieved and verified against `journal/results/tinyperson_fcos_factorial_seed42_download_manifest.json`.
Strict reload on CUDA confirmed 32,064,455 parameters (0 missing/unexpected keys).
Independent full-validation replay on NVIDIA GeForce RTX 5070 Ti across all 118 validation originals (1,684 tiles) achieved `FACTORIAL_REPLAY_ACCEPTANCE=PASS` with $\Delta \le 4.77 \times 10^{-7} \ll 0.0005$.
Factorial ANOVA and 10,000 bootstrap resamplings established:
- Top arm `l1c1s0` (H-WIoU + Gaussian Centerness): $\text{AP}_{50} = \mathbf{0.3113}$ (+3.61% over baseline $0.2752$), and $\text{AP}^{\text{tiny1}} = \mathbf{0.1200}$ (+115% surge over baseline $0.0558$).
- Negative controls `l0c1s0` and `l0c1s1` (GIoU + Gaussian Centerness) completely collapsed to $\text{AP} = 0.0000$.
- Giant positive interaction effect: $\Delta_{\text{Loss} \times \text{Ctr}} = +0.1580$, $p = 1.43 \times 10^{-24}$.
See `journal/audits/hwiou_seed42_fcos_tinyperson_factorial_postrun_2026-09-11.md`.

## TinyPerson Faster R-CNN Placement Factorial Gate (Revision v4, 2026-09-11)

The four-arm Faster R-CNN placement implementation is decoupled and Journal-only. Revision v4 resolved the inlined evaluator alias bug (`TinyPersonParams`), passing all 7/7 unit tests in `scripts/test_tinyperson_frcnn_placement.py`.
Under explicit user authorization on 2026-09-11 ("1. đồng ý"), fresh live preflight certified >4.2h GPU quota across four idle accounts (`hienquang06`, `dipphmngc`, `hngngnguynvn`, `trieuvo123`), the immutable pre-run was frozen in `journal/audits/hwiou_seed42_frcnn_placement_prerun_v4_2026-09-11.json`, and all four arms were dispatched to Kaggle:
1. `hienquang06/tod-tp-frcnn-place-standard-s42`: `KernelWorkerStatus.RUNNING`
2. `dipphmngc/tod-tp-frcnn-place-loss-only-s42`: `KernelWorkerStatus.RUNNING`
3. `hngngnguynvn/tod-tp-frcnn-place-assignment-only-s42`: `KernelWorkerStatus.RUNNING`
4. `trieuvo123/tod-tp-frcnn-place-dual-s42`: `KernelWorkerStatus.RUNNING`
Seven consecutive polls across a 300-second window confirmed healthy execution without initialization error.
Dispatch receipt: `journal/audits/dispatch_receipts/receipt_frcnn_placement_2026-09-11_07-44-25_SUCCESS.json`.
Governing record: `journal/audits/hwiou_seed42_frcnn_placement_dispatch_v4_2026-09-11.md`.

## Current AI-TOD-v2 FCOS Gate (Revision v2r7, 2026-09-11)

Protocol `aitod_fcos_loss_only_v2` packaging revision v2r7 supersedes v2r6 by adapting the schedule to 10 epochs (~10.2h estimated runtime, guaranteeing completion within the Kaggle 12-hour limit).
Local candidate notebooks are generated in `.runtime/kaggle_aitod_fcos_loss_only_seed42/`, and all 6/6 local protocol identity gates certified PASS (`scripts/launch_aitodv2_fcos_cluster.py --verify-only`).
The design audit is frozen in `journal/audits/hwiou_seed42_fcos_aitodv2_design_v2r7_2026-09-11.json`.
A serialized capacity audit of all 13 accounts established that the private AI-TOD-v2 dataset is isolated to accounts `ngquangnht` and `phuc1806`, whose weekly GPU quota is exhausted and resets at `2026-09-12 00:00:00 UTC` (~16 hours). Candidates and configs are fully certified and ready for dispatch immediately upon quota refresh.


## Official-Test Results Update (2026-09-25)

The manuscript now reports 71 unique Seed-42 official-test evaluations: 20 canonical models and 51 extended models. This includes 32 AI-TOD-v2 models and 39 TinyPerson models. The canonical audit verifies 20/20; extended post-run evidence and completed-run records account for the other 51. The row-level evidence and metric reconciliation are in `journal/audits/official_test_manuscript_reconciliation_20260925.json`.

Validation scores remain labeled as validation and appear separately from official-test scores. The test results are single-seed, model-specific observations; they do not establish a universal method advantage. No further test inference is authorized by this status note.
