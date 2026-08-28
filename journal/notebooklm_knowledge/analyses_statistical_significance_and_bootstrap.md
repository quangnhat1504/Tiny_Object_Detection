---
title: "Statistical Significance Testing & Bootstrap Confidence Intervals"
type: "analysis"
created: "2026-08-23"
updated: "2026-08-23"
sources:
  - "journal/manuscript/main.tex"
  - "journal/tools/compute_statistics.py"
tags:
  - "statistics"
  - "hypothesis-testing"
  - "bootstrap"
---

# Statistical Significance Testing & Bootstrap Confidence Intervals

## 1. Methodology
To prevent reporting lucky seed variations, we implement rigorous hypothesis testing across 16 partition folds on the validation distribution.

## 2. Hypothesis Testing Results
1. **Paired Student's $t$-test**:
   * Null Hypothesis $H_0$: $\mu_{\text{H-WIoU}} = \mu_{\text{Baseline}}$
   * Alternative Hypothesis $H_1$: $\mu_{\text{H-WIoU}} > \mu_{\text{Baseline}}$
   * Computed $t$-statistic: **$t = 73.1805$**
   * Two-tailed $p$-value: **$p = 1.42 \times 10^{-20} \ll 0.0001$**
   * Conclusion: Reject $H_0$ with overwhelming statistical certainty.

2. **Wilcoxon Signed-Rank Test (Non-Parametric)**:
   * Test statistic: **$W = 0.0$**
   * $p$-value: **$p = 4.38 \times 10^{-4} < 0.001$**
   * Conclusion: H-WIoU strictly dominates baseline across 100% of tested partition splits.

## 3. Non-Parametric Bootstrap Confidence Intervals ($N = 10,000$)
Using $N = 10,000$ empirical Monte Carlo resamplings:
* **Baseline $\text{mAP}_{50}$ 95% CI**: $[0.4002, 0.4042]$
* **H-WIoU $\text{mAP}_{50}$ 95% CI**: $[0.4596, 0.4628]$
* **Empirical Gain $\Delta(\text{Gain})$ 95% CI**: $[+0.0574, +0.0605]$

The complete separation between the upper bound of Baseline ($0.4042$) and the lower bound of H-WIoU ($0.4596$) establishes definitive, uncompromised empirical superiority.
