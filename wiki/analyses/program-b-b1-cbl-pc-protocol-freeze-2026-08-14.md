---
title: Program B B1 CBL/PC Protocol Freeze - 2026-08-14
type: analysis
created: 2026-08-14
status: B1_READY_FOR_OWNER_APPROVAL_NO_TRAINING
sources:
  - wiki/syntheses/strategic-research-roadmap-2026-08-14.md
  - wiki/syntheses/program-b-cbl-pivot-decision-2026-08-14.md
  - wiki/analyses/iterative-cbl-fair20-locked-test-protocol-2026-08-01.md
  - .runtime/kaggle/cbl_iterative_train_fair20/state.json
  - .runtime/kaggle/cbl_iterative_train_fair20/output/tod_output/protocol.json
  - .runtime/kaggle/pc_micro_fair20_seed42/state.json
  - scripts/test_pc_mr_moc_configuration.py
  - scripts/test_pc_mr_moc_combination.py
tags: [program-b, cbl, pc-mr, pc-moc, protocol, baseline-freeze, no-training]
---

# Program B B1 CBL/PC Protocol Freeze - 2026-08-14

## Status and authorization boundary

This was the owner-reviewable B1 package. The original un-tiled surface failed
the [[Program B B1 Scale-Match Audit - 2026-08-14]]. The subsequent tested
[[Program B B1 Tiled Scale Revision Audit - 2026-08-14]] restores baseline
tiling and preserves original IDs; its scale contract passes. The manifest-backed
original-image evaluator path now passes its integration gate in
[[Program B B1 Evaluator-Integration Gate - 2026-08-14]]. B1 is
`READY_FOR_OWNER_APPROVAL_NO_TRAINING`; it does **not** authorize a Kaggle push,
training, a historical locked-test
evaluation, or a performance claim.

Paper A remains closed `NO-GO`. The historical 65-image CBL locked-test budget
is consumed and excluded from every Program B arm. Historical fair-20 PC-MR,
PC-MOC, RA-TB, and CR-SC artifacts remain diagnostic-only; no historical score
is a B1 baseline or a promotion result.

## Research question

Can one of the two teacher-bounded mechanisms below improve strict
micro-object localization beyond a frozen iterative-CBL baseline, while leaving
the trained student's inference graph teacher-free?

1. **PC-MR-RPN**: teacher-bounded micro-object proposal rescue, whose auxiliary
   gradients affect only the student RPN head.
2. **PC-MOC-FD**: teacher-bounded micro-object feature distillation, whose
   auxiliary gradients affect only the student FPN.

The first comparison is baseline versus each individual candidate. A
PC-MR+PC-MOC arm is prohibited until both individual candidates meet the full
validation-promotion rule. RA-TB, PC-MHFD, and all historical CBL branches are
out of scope.

## Semantic baseline reference

The reference is the completed iterative-CBL fair-20 artifact, retained only as
a semantic and teacher reference:

| Field | Frozen reference |
|---|---|
| Source commit | `80e934aaa7555733d795a8adbe70c19027e67735` |
| Artifact path | `.runtime/kaggle/cbl_iterative_train_fair20/output/tod_output/runs/sa_alw_full__cbl__irtw0.5ir1s0.3__la_loss__seed42__cbl_iterative_train_fair20/best.pt` |
| Artifact SHA-256 | `90043edfd278a51eef76c8494f4edae8e37127e78fc79dda9eee8071cc29769a` |
| Training budget | 20 epochs, no early stop |
| Student transform | `640/800` |
| Assignment and localization | `sa_alw_full`, `la_loss`, CBL |
| CBL refinement | train weight `0.5`; one train pass; one inference pass; blend `1.0`; score threshold `0.30` |
| CBL grid | alpha `5.0`, bins `6`, grid beta `1.0`, uncertainty-match weight `1.0` |
| Optimizer/schedule | SGD, two-epoch warmup, 20-epoch cosine; EMA decay `0.9998` |
| Checkpoint selector | validation `mAP_50`, `best.pt` |

The old artifact's test metrics and the historical locked-test data are not
part of the Program B evaluator or selection rule.

## Execution source freeze

The semantic baseline commit does not contain the current PC-MR/PC-MOC surface,
and the checkout has baseline-critical working-tree changes. Per the approved
no-commit route, B1 freezes a content-addressed source bundle rather than
claiming that the mutable checkout is a commit-level execution identity.

| Field | Frozen B1 execution reference |
|---|---|
| Git context only | `113ce299341413bb740c7aeec16319975ac3d14e` on `cbl-rpn-iou-quality-20260731` |
| Source bundle | `.runtime/local/program_b/b1_source_20260814/program_b_source_bundle.zip` |
| Bundle SHA-256 | `4088ca39c7eeab5fb278a0558fb53457d94becb749ecdfc384baf8998e43555d` |
| Source manifest | `program_b_source_manifest.json`, 18 curated source/test/dependency files |
| Environment lock | `program_b_environment_lock.json`, SHA-256 `1660ec4c9742913b5437d4bfd26d36479923ceadd7794f8aafbf4bcecd947b66` |
| Runtime | Python `3.13.9`, PyTorch `2.11.0+cu128`, one RTX 5070 Ti |

The manifest records per-file SHA-256 values for the baseline, PC mechanisms,
split builder, evaluator bridge, technical tests, and dependency declaration.
Bundle integrity was independently rechecked after creation. Every B2 arm must
unpack/use this exact source bundle and use only declared command-line flags for
the baseline/candidate difference.

## Data and original-image evaluator freeze

The legacy `YOLOTinyDataset` tile index preserves original-image identity, but
`paper_a/splits/split_audit.json` establishes that the current derivative is
`NO_GO_CURRENT_DERIVATIVE`: video/sequence overlap is `30` train/validation,
`23` train/test, and `20` validation/test groups. It also lacks upstream
original paths and ignore/crowd provenance, and its evaluator treats tiles as
independent COCO images. It cannot be a Program B training, validation, or
external-test surface.

Program B instead uses the official TinyPerson train-side original-image task
annotation, with a distinct namespace-derived validation split:

| Field | Frozen B1 data reference |
|---|---|
| Source annotation | `D:/paper_a_data/TinyPerson/tiny_set/erase_with_uncertain_dataset/annotations/task/tiny_set_train_all.json` |
| Source SHA-256 | `c4f5bef58be3bd7b6b622a4a2ac7030255d2321ef5e59dcc2d153b999e0de407` |
| Split manifest | `.runtime/local/program_b/b1_tinyperson_split_20260814/program_b_split_manifest.json` |
| Split SHA-256 | `c3d741aa4178316bf0fcc6cdb50e1cb78643d2d6c964c12f3247a38296b4bb2d` |
| Group rule/order | video stem without `_I<frame>`, otherwise image stem; `sha256(program_b_b1_20260814:kind:identity)` |
| Train / validation | 628 / 118 original images; 16,193 / 2,240 annotations |
| Group overlap | none (`[]`) |
| Train / validation annotation SHA-256 | `9c3f76ceca8c60f43d75e519a1db7c22e060e9f6cfb20fbe36076a062a9e91cc` / `dfa4d92b4b37bdf257e47357efe7cc502b1e97cd10178e8864b722adcfea645d` |

Use `scripts/build_program_b_tinyperson_split.py` only to regenerate and verify
the exact manifest, never to replace it after B2 begins. The original-image
dataset adapter successfully loaded a validation image at native `1080x1920`
resolution. The pinned TinyPerson official evaluator bridge verified its source
SHA-256 `222b3173510e7a89bd03d077dce5d4a11e23ea6a7cd22afbbe930817b0886557`.
The new scale audit supersedes this as an execution-ready freeze: after the
same `640/800` model transform, its training median square-root box area is
`5.63 px` versus `13.53 px` for the sampler-weighted iterative-CBL tile
surface. This split remains a valid original-image provenance artifact, but is
not a B2 training surface until a tested Program B tiling adapter restores the
predeclared scale contract.
For tiled inference, `paper_a/evaluation/tile_to_original.py` must reconstruct
predictions and GT on original images before official COCO accumulation. Its
fixture proves:

- no source/video group crosses train and validation;
- each original image is evaluated once, despite tiling;
- empty-GT and empty-prediction original images are retained;
- AP, AP50, AP75, and AR100 use a documented COCO configuration; and
- deterministic source-group folds are reproducible from the manifest.

No external test is mounted or read for B2/B3. Before B4, the owner must name a
separate external source and freeze its manifest/hash while keeping it unread
through all training and validation. No historical locked-test or Paper A test
surface may substitute for it.

## Frozen candidate configurations

Both candidates load the same frozen `best.pt` teacher above, verify its
SHA-256 before model construction, run it at `960/1200` only while training,
set `eval()`, set every teacher parameter `requires_grad=False`, and exclude
teacher state from student checkpoints and inference.

| Setting | PC-MR-RPN | PC-MOC-FD |
|---|---:|---:|
| Loss weight | `0.005` | `0.15` |
| Teacher transform | `960/1200` | `960/1200` |
| Proposal top-N | `300` | `300` |
| Micro cutoff | `8.0 px` | `8.0 px` |
| Teacher IoU floor | `0.50` | `0.50` |
| Teacher advantage margin | `0.02` | `0.02` |
| Feature target | n/a | `cosine` |
| Auxiliary scope | RPN head only | FPN only |
| Gradient handling | disjoint PCGrad | disjoint PCGrad |

The combined arm, if later eligible, must reuse one identical teacher object,
identical selection settings, and disjoint RPN-head/FPN PCGrad scopes. It is
not an authorized B2 arm.

## Planned validation matrix and measurement

Only after the scale/adaptor revision, fresh B1 approval, and a B2 pre-run
report may the following 20-epoch matched matrix run on seeds `42`, `123`, and
`2024`:

| Arm | B2 status |
|---|---|
| Iterative-CBL baseline | required |
| PC-MR-RPN | required, individually paired with baseline |
| PC-MOC-FD | required, individually paired with baseline |
| PC-MR-RPN + PC-MOC-FD | prohibited pending both individual promotions |

All three B2 arms must share data manifest, ordered sampler, augmentation,
student transform, optimizer/schedule, EMA policy, hardware class, checkpoint
selector, evaluator, source snapshot, and artifact contract. Selector choice is
frozen before launch and cannot be changed after metrics are observed.

Report AP and AP75 as primary endpoints. Also report AP50, AR100,
class-aware micro/tiny AP, original-image source-group folds, latency, peak
VRAM, parameter count, training time, checkpoint hashes, and full prediction
artifacts. The primary comparison uses matched original-image paired bootstrap
of candidate minus baseline, with shared resamples within each fixed seed; it
must record resampling unit, replicate count, RNG seed, confidence-interval
method, and evaluator version.

## Promotion rule

An individual candidate is eligible for a B2 `GO` only when every condition is
true:

1. the complete artifact package and independent reload pass on the frozen
   source snapshot;
2. the three-seed mean AP delta versus the exact baseline is positive;
3. the paired original-image AP interval supports a positive primary effect;
4. AP75 and class-aware micro/tiny diagnostics show no material contradiction;
5. at least two deterministic source/image folds do not reverse the primary
   conclusion; and
6. latency, VRAM, parameter count, and train time are recorded.

A completed kernel, a one-seed gain, AR-only improvement, an unpaired baseline,
or a legacy fair-20 score cannot satisfy this rule.

## B1 local technical gates

The gates below establish integration integrity only; they are not performance
evidence and do not authorize training.

| Gate | Required evidence | Current result |
|---|---|---|
| CLI configuration contract | `scripts/test_pc_mr_moc_configuration.py` | PASS on 2026-08-14 |
| Syntax/import surface | `py_compile` of model, training, and PC test/audit modules | PASS on 2026-08-14 |
| Teacher identity | local SHA-256 of reference `best.pt` | PASS: `90043edfd278a51eef76c8494f4edae8e37127e78fc79dda9eee8071cc29769a` |
| CUDA availability | CUDA runtime/device query | PASS: PyTorch `2.11.0+cu128`, one RTX 5070 Ti |
| Real combined forward/backward/reload smoke | `scripts/test_pc_mr_moc_combination.py`, four valid optimizer steps | PASS: 4/4 steps; JSON `9474393125c34caaf5f580c9bb3a69bbc446ef2dc753e046bd56afc847c100c2`, checkpoint `cc5f67ca9b2e2b9f0dcf67ff4b20c065d9f22e8ac7f19992fedd8b04c34b37d4` at `.runtime/local/program_b/` |
| Teacher-free inference, finite gradients, scope assertions | enforced by the real smoke | PASS: inference/reload outputs exact; teacher state absent; 0 teacher gradient parameters |
| Program B split contract | unit test plus generated manifest | PASS: 628/118 originals, zero source-group overlap |
| Original-image evaluator bridge | 9 tile-to-original fixture tests plus a real original-image load | PASS: 10 tests total; native `1080x1920` validation image loaded |
| Official evaluator source pin | pinned TinyPerson evaluator SHA-256 | PASS: `222b3173510e7a89bd03d077dce5d4a11e23ea6a7cd22afbbe930817b0886557` |
| Execution source bundle | manifest/file/archive SHA-256 verification | PASS: 18 files; bundle SHA-256 matches manifest |

## Owner decision required

The source bundle, grouped original-image validation split, evaluator bridge,
and local technical gates are retained as evidence, but the scale-match audit
supersedes the former owner-ready status. Do not approve B1 or authorize B2
until a replacement tiled Program B data adapter, its regenerated source bundle,
and a passing scale audit are reviewed. The later B2 pre-run report must repeat
the replacement source/data hashes, list exact commands, confirm that no
external-test asset is mounted, and obtain explicit training authorization.

## Related pages

- [[Program B CBL Pivot Decision — 2026-08-14]]
- [[Strategic Research Roadmap — 2026-08-14]]
- [[Iterative CBL Fair-20 Locked-Test Protocol - 2026-08-01]]
- [[PC Micro Fair-20 Protocol - 2026-08-02]]
- [[Wiki Overview]]
- [[Wiki Log]]
