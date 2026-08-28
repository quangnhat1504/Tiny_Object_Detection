---
title: Wiki Index
type: overview
created: 2026-05-09
updated: 2026-08-10
sources: []
tags: [system]
---

## Wiki Index

<!-- markdownlint-disable MD060 -->

## Overview

- [[Wiki Overview]]
- [[Wiki Log]]

## Sources

| Page | Summary | Date | Tags |
|------|---------|------|------|
| [[ALW]] | Proposal for an anisotropic log-Wasserstein metric for tiny object detection. | 2026-05-09 | #tiny-object-detection #metric #loss #wasserstein |
| [[NWD]] | Normalized Gaussian Wasserstein Distance for IoU-insensitive tiny object assignment/loss/NMS. | 2026-05-31 | #tiny-object-detection #metric #wasserstein |
| [[GCD]] | Gaussian Combined Distance with scale-invariant, coupled box geometry terms. | 2026-05-31 | #tiny-object-detection #metric #wasserstein |
| [[IGWD Paper]] | Improved Gaussian Wasserstein Distance for smooth, scale-invariant tiny object detection. | 2026-05-31 | #tiny-object-detection #metric #wasserstein |
| [[RFLA]] | Gaussian receptive-field based label assignment for tiny object detection. | 2026-05-31 | #tiny-object-detection #label-assignment |
| [[Tiny Object Metrics Comparison Filled]] | Filled local results table for NWD, GCD, IGWD, and ALW experiments. | 2026-05-31 | #experiment-results #metrics |
| [[SAH-GD Hybrid Metrics Comparison]] | Ablation results for four SAH-GD variants: ADAPTIVE_NWD, HARD_SWITCH, SOFT_BLEND, SCALE_TOPK. | 2026-05-31 | #experiment-results #sah-gd #ablation |
| [[P2 Feature Implementation - 2026-05-31]] | Implementation of P2 (stride-4) FPN level for micro object detection (<8px). | 2026-05-31 | #p2-features #architecture #implementation |
| [[RG-Robust ALW Implementation]] | Merged ALW with dynamic top-k and reliability-gated robust shape; why the reference run won. | 2026-06-02 | #alw #label-assignment #implementation |
| [[Bảng Tổng Hợp Kết Quả Notebooks 1-12]] | Aggregated results from all 12 Kaggle notebooks. | 2026-06-05 | #experiment-results #kaggle |
| [[ALW Main Draft]] | Current manuscript draft for the ALW full paper and its open evidence gaps. | 2026-06-10 | #alw #paper-draft |
| [[Paper A SA-ALW Conference Refinement Plan]] | Frozen scope, evidence gates, public-benchmark protocol, and submission criteria for Paper A. | 2026-08-02 | #sa-alw #paper-a #protocol |
| [[WP03 Canonicality and Matching Audit — 2026-08-14]] | A1 artifact/code audit: canonical implementation passes; no promotion or re-adjudication without owner approval. | 2026-08-14 | #sa-alw #paper-a #audit |
| [[WP03 A2 T4 Re-adjudication Amendment — 2026-08-14]] | Frozen PL-003 decision and its immutable v12 T4 reproducibility evidence. | 2026-08-14 | #sa-alw #paper-a #protocol |
| [[WP03 A3 Seed-42 Pre-Run — 2026-08-14]] | The two exact seed-42 packages, held pending verified GPU capacity. | 2026-08-14 | #sa-alw #paper-a #kaggle |
| [[WP03 A4 Paper A NO-GO — 2026-08-14]] | Complete matched matrix and paired bootstrap close Paper A performance work NO-GO. | 2026-08-14 | #sa-alw #paper-a #decision |

## Entities

| Page | Type | Summary |
|------|------|---------|

## Concepts

| Page | Summary | Source Count |
|------|---------|--------------|
| [[Anisotropic Log-Wasserstein Distance (ALW)]] | Metric that combines Wasserstein-style position terms with log-ratio shape terms. | 2 |
| [[IGWD]] | Improved Gaussian Wasserstein Distance concept and baseline analyzed by ALW. | 3 |
| [[Scale-Adaptive Hybrid Gaussian Distance (SAH-GD)]] | Proposed adaptive hybrid distance blending NWD and GCD by object scale. | 2 |
| [[Scale-Adaptive Anisotropic Log-Wasserstein Distance (SA-ALW)]] | Target-conditioned ALW extension with clipped non-learned beta and position schedules. | 3 |
| [[Cascaded Uncertainty Routing]] | YOLO→Uncertainty Router→FRCNN(patch)→WBF cascaded architecture. | 1 |

## Topics

| Page | Summary | Source Count |
|------|---------|--------------|
| [[Tiny Object Detection Metrics]] | Summary of metric design issues, local experiment results, and proposed fixes for tiny object detection. | 8 |
| [[CIoU/DIoU AMP Crash Fix for Tiny Boxes]] | Fix for CUDA kernel crash when training CIoU/DIoU box loss under AMP float16 on tiny boxes. | 1 |

## Analyses

| Page | Question | Date |
|------|----------|------|
| [[Kaggle T4 Run Handoff — 2026-07-13]] | Historical T4 kernel handoff plus 2026-07-23 artifact-handling update. | 2026-07-13 |
| [[Tiny Object Metric Experiment - 2026-05-31]] | What should be improved after the 4 metric runs underperformed expectations? | 2026-05-31 |
| [[Tiny Object Metric Ablation Plan - 2026-05-31]] | How should SAH-GD be tested on Kaggle before deeper architecture changes? | 2026-05-31 |
| [[Tiny Object Architecture Improvement - 2026-05-31]] | After HARD_SWITCH wins, what architectural changes should be prioritized? | 2026-05-31 |
| [[P2 Experiment Result - 2026-06-02]] | Did P2 improve micro detection, is it a net win, and what next? | 2026-06-02 |
| [[SAH-GD Advancement - 2026-06-02]] | Can the SAH-GD line progress further or has metric innovation plateaued? | 2026-06-02 |
| [[ALW Failed vs Success Comparison]] | Why did the original ALW fail but the reference run succeed? | 2026-06-05 |
| [[ALW vs IGWD Comparison]] | Comparative analysis of ALW and IGWD metric performance. | 2026-06-05 |
| [[Action Plan: Post Deep-Research Execution Roadmap]] | What to do after the deep research workflow completed? | 2026-06-05 |
| [[ALW Paper Improvement Task List - 2026-06-10]] | What tasks are needed to improve the ALW paper? | 2026-06-10 |
| [[ALW Full Paper Action Plan - 2026-06-10]] | What is missing to turn the ALW draft into a defensible full paper? | 2026-06-10 |
| [[COCO Metrics Migration Plan - 2026-06-12]] | How to migrate evaluation from scale-aware to COCO standard metrics? | 2026-06-12 |
| [[Cascaded Routing Implementation Plan - 2026-07-01]] | How to implement the cascaded YOLO→FRCNN→WBF architecture with uncertainty routing? | 2026-07-01 |
| [[Phase 0 Dataset Statistics - 2026-07-01]] | What are the data statistics and variance analysis on SOD-TinyPeopleInSea? | 2026-07-01 |
| [[Phase 1 Baseline Setup - 2026-07-01]] | What are the 3 independent baselines for the cascaded routing experiment? | 2026-07-01 |
| [[Phase 2 Metric Chain Ablation - 2026-07-01]] | What is the contribution of each metric in the IoU→NWD→IGWD→ALW→SA-ALW chain? | 2026-07-01 |
| [[Phase 2-4 Results Summary]] | What are the complete results across Phase 2 (metrics), Phase 3 (assigner), and Phase 4 (cascade)? | 2026-07-04 |
| [[Test-Set Evaluation — Phase 2 Metrics]] | How do Phase 2 metrics perform on the locked test set? | 2026-07-04 |
| [[WBF Improvement — Root Cause Analysis & Plan]] | Why does WBF hurt large-object AP and how to fix it? | 2026-07-04 |
| [[Paper Rewrite Summary - 2026-07-06]] | What was changed in the ALW and SA-ALW paper rewrite? | 2026-07-06 |
| [[Deep Research: Tiny-OD Breakthroughs 2024–2026 & the AP@75 Diagnosis]] | Which 2024–2026 methods can break our AP@75 ceiling, and why is it really stuck? | 2026-07-06 |
| [[Decoupled DFL Regression Plan - 2026-07-06]] | How to break AP@75 by decoupling SA-ALW assignment from a DFL regression head? | 2026-07-06 |
| [[Decoupled Regression Breakthrough Plan - 2026-07-07]] | Experimental plan: SA-ALW assignment + box loss variants (Smooth-L1, CIoU, DIoU). | 2026-07-07 |
| [[CIoU/DIoU Decoupled Regression Training Failure — 2026-07-08]] | Why CIoU/DIoU training crashes under AMP float16 on tiny boxes. | 2026-07-08 |
| [[Confidence-Driven Localization Local Gate - 2026-07-30]] | Does distributional RoI localization pass the local AP75 and stability gate? | 2026-07-30 |
| [[CBL Quality Focal Loss Local Gate - 2026-07-30]] | Does a QFL joint class-IoU score improve CBL detection ranking? | 2026-07-30 |
| [[CBL Rank and Sort Local Gate - 2026-07-30]] | Does sampled RoI Rank & Sort improve CBL score-IoU ranking? | 2026-07-30 |
| [[CBL EMA Recovery Audit - 2026-07-30]] | Is the recovered epoch-5 EMA peak exact, reloadable, and promotion-ready? | 2026-07-30 |
| [[Double-Head CBL Local Gate - 2026-07-31]] | Does a dedicated convolutional RoI regression branch improve CBL localization? | 2026-07-31 |
| [[Iterative CBL Refinement Gate - 2026-07-31]] | Does inference-time repeated CBL regression improve strict localization? | 2026-07-31 |
| [[Trainable Iterative CBL Local Gate - 2026-07-31]] | Can shared-head refined-proposal supervision improve CBL localization without a full cascade? | 2026-07-31 |
| [[Stage-Specific CBL Refinement Local Gate - 2026-07-31]] | Does a separate second-pass CBL regressor beat shared-head iterative refinement? | 2026-07-31 |
| [[CBL Cascade Stage-2 Local Gate - 2026-07-31]] | Can higher-IoU stage-2 re-matching and classification reduce iterative CBL background errors? | 2026-07-31 |
| [[CBL Refinement Consistency and Depth Gate - 2026-07-31]] | Does refinement consistency predict localization quality, and how many passes should the trainable CBL leader use? | 2026-07-31 |
| [[Unrolled Iterative CBL Training Local Gate - 2026-07-31]] | Does matching the three-pass inference trajectory during shared-head training improve CBL localization? | 2026-07-31 |
| [[CBL Refinement Trajectory and Damped Final Step - 2026-07-31]] | Can cross-fit pass selection or a damped final update improve the fixed CBL refinement trajectory? | 2026-07-31 |
| [[CBL Horizontal-Flip TTA Local Gate - 2026-07-31]] | Can paired original/flip box and score fusion improve the fixed CBL leader? | 2026-07-31 |
| [[CBL Transform-Scale TTA Local Gate - 2026-07-31]] | Can transform-scale TTA and size-aware pair calibration improve fixed CBL inference beyond flip TTA? | 2026-07-31 |
| [[CBL Stochastic Multi-Scale Training Local Gate - 2026-07-31]] | Can stochastic training scales absorb the scale-TTA gain into CBL weights? | 2026-07-31 |
| [[CBL SNIP-Like Scale-Normalized Training Local Gate - 2026-07-31]] | Can ignored-object scale supervision recover stochastic multi-scale CBL training? | 2026-07-31 |
| [[CBL RPN Proposal Coverage Audit - 2026-07-31]] | Is high-IoU proposal coverage the next micro-object localization bottleneck? | 2026-07-31 |
| [[CBL Iterative RPN Refinement Gate - 2026-07-31]] | Can repeated fixed RPN deltas convert proposal IoU75 gains into detector AP75? | 2026-07-31 |
| [[CBL RPN Quality Objectness Local Gate - 2026-07-31]] | Can IoU-aware RPN objectness improve strict proposal ranking without losing micro objects? | 2026-07-31 |
| [[CBL Learned RPN Cascade Local Gate - 2026-07-31]] | Can detached coarse-to-fine RPN refinement and re-matching improve strict proposal localization? | 2026-07-31 |
| [[CBL RPN IoU Quality EMA8 Audit - 2026-08-01]] | Can separate presence and localization-quality RPN signals beat the iterative-CBL leader over a full eight-epoch EMA schedule? | 2026-08-01 |
| [[Iterative CBL Fair-20 Locked-Test Protocol - 2026-08-01]] | Completed fair 20-epoch iterative-CBL run and one preregistered locked test; new leader AP/AP50/AP75/AR100=`0.1158/0.3326/0.0533/0.2657`. | 2026-08-01 |
| [[Cross-Scale CBL Localization Distillation Plan - 2026-08-01]] | Validation-only SC-CBL candidate: transfer advantage-gated `960/1200` teacher coordinate distributions into the fixed-scale student with no inference change. | 2026-08-01 |
| [[Conflict-Aware SC-CBL Plan - 2026-08-01]] | Next method: preserve shared cross-scale adaptation while projecting auxiliary gradients that conflict with the base detector objective. | 2026-08-01 |
| [[Coordinate-Reliable SC-CBL Plan - 2026-08-01]] | Per-coordinate reliable cross-scale CBL distillation passed fresh-seed local gates; its private fair-20 validation run is in progress. | 2026-08-01 |
| [[CR-SC-CBL Multi-Seed Fair-20 Protocol - 2026-08-02]] | Frozen three-seed paired fair-20 validation matrix across five private Kaggle runs, with exact teacher/source hashes and artifact-first gates. | 2026-08-02 |
| [[Post-CR-SC-CBL Mechanism Gates - 2026-08-02]] | Rejects four mechanisms, tracks RA/RA-TB performance gates, and freezes the evidence-backed micro-rescue RPN pivot. | 2026-08-02 |
| [[RA-TB-CBL Fair-20 Protocol - 2026-08-02]] | Frozen same-source seed-42 20-epoch EMA baseline/candidate validation pair, exact smokes, and artifact-first promotion gate. | 2026-08-02 |
| [[PC-MOC-FD Gates - 2026-08-02]] | Teacher-bounded micro-object FPN distillation passed Gate0, CUDA, and robust seed2718 performance gates; fair-20 is running. | 2026-08-02 |
| [[PC-MHFD Gates - 2026-08-02]] | High-frequency micro-object FPN distillation improved aggregate AP/AP75/AR but failed fold and micro/tiny robustness gates; rejected. | 2026-08-02 |
| [[RA-TB plus PC-MHFD Combination Gates - 2026-08-02]] | Compatibility and CUDA gates passed, but the independent PC-MHFD prerequisite failed, so no combination run is launched. | 2026-08-02 |
| [[PC Micro Fair-20 Protocol - 2026-08-02]] | Shared seed42 20-epoch EMA validation matrix for promoted PC-MOC and PC-MR with one same-source baseline and frozen auditors. | 2026-08-02 |
| [[Program B B1 CBL/PC Protocol Freeze - 2026-08-14]] | Baseline geometry, source-group-disjoint tiled data, and manifest-backed original-image evaluation are integrated; B2 is blocked only by the incompatible Kaggle P100 hardware assignment. | 2026-08-17 |
| [[Program B B1 Scale-Match Audit - 2026-08-14]] | REVISE: measured post-transform object scale differs materially from iterative-CBL tiled training; requires a tested COCO-to-tile adapter before B1/B2 approval. | 2026-08-14 |
| [[Program B B1 Tiled Scale Revision Audit - 2026-08-14]] | Tested `512/64` COCO-to-tile revision restores geometry and passes the frozen scale contract; later manifest-backed evaluator integration passed. | 2026-08-17 |
| [[Program B B2 Authorized Train-from-Scratch Protocol - 2026-08-14]] | B2 evaluator-integrated source snapshot is ready, but the required mount smoke was assigned incompatible P100 hardware; baseline and candidates are blocked with no training evidence. | 2026-08-17 |
| [[Program B B2 Baseline v4 Recovery Audit - 2026-08-17]] | Recovered a completed pre-protocol baseline artifact; it is diagnostic-only because its source/evaluator do not satisfy the frozen B2 contract. | 2026-08-17 |
| [[Maximum-Performance Research Checkpoint - 2026-08-02]] | Goal milestone and paper handoff: frozen leader, promoted/rejected methods, PC-MR plus PC-MOC compatibility evidence, active fair-20 jobs, and claim boundary. | 2026-08-02 |
| [[SA-ALW Paper Refinement Phase 0-2 - 2026-08-02]] | Canonical method/protocol, public evaluator fixtures, test-access disclosure, atomic Kaggle team shards, result ledgers, and compiled manuscript. | 2026-08-02 |
| [SA-ALW Paper Resume Checkpoint - 2026-08-02](analyses/sa-alw-paper-resume-checkpoint-2026-08-02.md) | End-of-day Paper A gate state, mechanism evidence, schedule/pilot freeze, Kaggle boundary, and exact resume order. | 2026-08-02 |
| [[TinyPerson Acquisition and Real-Data Fixture - 2026-08-04]] | Is the official TinyPerson package acquired and verified, and does the Paper A fixture chain run on real data? | 2026-08-04 |
| [[TinyPerson G1 Bounds, Mechanism Diagnostics, and Validation Proposal - 2026-08-04]] | What are TinyPerson's train-only scale bounds, does the SA-ALW mechanism reproduce there, and how to handle its missing validation split? | 2026-08-04 |
| [[TinyPerson Pilot Harness and Pre-Run Freeze - 2026-08-04]] | Does the WP01 TinyPerson pilot have a canonical harness, passing smokes, and a READY_FOR_PUSH pre-run report? | 2026-08-04 |
| [[Program B B1 Evaluator-Integration Gate - 2026-08-14]] | Evaluator integration validation and dataset manifest verification for Program B. | 2026-08-14 |
| [[Program B 3-Seed Multi-Arm Benchmark and Statistical Report]] | Statistical analysis across 3 random seeds (42, 123, 2024) on Tesla T4 for Iterative-CBL, PC-MR, PC-MOC, and Joint. | 2026-08-20 |
| [[21-Model 20-Epoch Mega-Benchmark Statistical Report]] | Complete 21-model benchmark matrix (7 methods x 3 seeds) on TinyPerson b1-tiled under 20-epoch budget. | 2026-08-20 |
| [[AI-TOD-v2 SOTA Benchmark & Per-Class Comparative Report]] | Comprehensive SOTA evaluation and per-class breakdown on AI-TOD-v2 test set for H-WIoU and competing methods. | 2026-08-23 |
| [[AI-TOD-v2 Full Empirical Benchmark Matrix]] | Live execution tracking of 7 concurrent Kaggle GPU runs on AI-TOD-v2 for Faster R-CNN, NWD, IGWD, RFLA, and H-WIoU variants. | 2026-08-23 |

## Research

| Page | Summary | Date |
|------|---------|------|
| [[Deep Research: Architecture & Training Strategies for Tiny Object Detection]] | Deep research into architecture and training strategies for tiny object detection. | 2026-06-05 |

## Syntheses

- [Strategic Research Roadmap — 2026-08-14](syntheses/strategic-research-roadmap-2026-08-14.md) — Evidence-gated Paper A decision and conditional CBL/PC pivot.
- [WP03 A3 Seed-42 Execution State — 2026-08-14](syntheses/wp03-a3-seed42-execution-2026-08-14.md) — Both matched seed-42 artifacts completed and passed package/reload gates before A4.
- [WP03 A4 Paper A NO-GO — 2026-08-14](syntheses/wp03-a4-paper-a-no-go-2026-08-14.md) — Complete matrix and paired bootstrap close Paper A performance work and open Program B B0 recovery.
- [Program B CBL Pivot Decision — 2026-08-14](syntheses/program-b-cbl-pivot-decision-2026-08-14.md) — Owner selects CBL/PC; B1 protocol/baseline freeze is the next package.
- [[Journal Project Memory Bank]] — Dedicated isolated memory bank for the H-WIoU Journal Project (IEEE TPAMI / IJCV) and empirical tracking.

<!-- markdownlint-enable MD060 -->
