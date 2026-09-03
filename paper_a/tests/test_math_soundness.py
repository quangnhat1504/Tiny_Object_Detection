"""
Rigorous Mathematical Soundness Test Suite for Scale-Adaptive Wasserstein-IoU (SA-WIoU / H-WIoU).

Covers all 8 formal mathematical properties:
1. Exact zero overlap -> finite similarity, finite loss, finite nonzero center gradient with correct direction (towards GT).
2. Symmetric center perturbations -> anti-symmetric gradient signs.
3. Exact box match -> S=1, L=0, zero gradient.
4. Scale asymptotic limits:
   - Large-target limit (s_B -> inf) -> IoU dominates.
   - Small-target regime (s_B -> 0) -> Transport divergence exp(-D_SN^2) dominates.
5. Numerical finite-difference gradient matches autograd gradient (relative error < 1e-3).
6. No NaN/Inf under degenerate-but-clamped dimensions or extreme coordinates.
7. Batch/vectorized similarity and loss match single-pair scalar computations (< 1e-6).
8. Compact domain separation lower bound: ||grad|| >= c > 0 for delta <= ||mu_a - mu_b||/s_B <= K.
"""
from __future__ import annotations
import math
import unittest
import torch

from common.metrics.h_wiou import (
    compute_scale_homotopy,
    pairwise_scale_normalized_divergence_squared,
    aligned_scale_normalized_divergence_squared,
    compute_h_wiou_similarity,
    aligned_h_wiou_loss,
)


class TestMathSoundnessHWIoU(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_1_zero_overlap_finite_and_correct_gradient_direction(self):
        """Property 1: Under strict zero overlap (IoU = 0), loss gradient must be nonzero
        and the gradient descent vector (-grad) must point strictly towards the ground truth centroid.
        """
        target = torch.tensor([[46.0, 46.0, 54.0, 54.0]], dtype=torch.float32)

        offsets = [
            (20.0, 0.0, 'right'),
            (-20.0, 0.0, 'left'),
            (0.0, 20.0, 'bottom'),
            (0.0, -20.0, 'top'),
            (15.0, 15.0, 'bottom-right'),
            (-15.0, -15.0, 'top-left'),
        ]

        for dx, dy, name in offsets:
            pred = torch.tensor([[46.0 + dx, 46.0 + dy, 54.0 + dx, 54.0 + dy]], dtype=torch.float32, requires_grad=True)

            loss = aligned_h_wiou_loss(pred, target, sigma_0=8.0)
            self.assertTrue(torch.isfinite(loss).all())
            self.assertGreater(loss.item(), 0.0)
            self.assertLess(loss.item(), 1.0)

            loss.backward()
            grad = pred.grad[0]

            grad_cx = (grad[0] + grad[2]).item()
            grad_cy = (grad[1] + grad[3]).item()

            dot_product = grad_cx * dx + grad_cy * dy
            self.assertGreater(dot_product, 0.0, f"Gradient direction failed for {name}: dot={dot_product}, grad=({grad_cx}, {grad_cy})")

    def test_2_symmetric_perturbations_opposing_gradients(self):
        """Property 2: Opposite center perturbations must produce exactly opposite centroid gradients."""
        target = torch.tensor([[46.0, 46.0, 54.0, 54.0]], dtype=torch.float32)

        d = 12.0
        pred_pos = torch.tensor([[46.0 + d, 46.0, 54.0 + d, 54.0]], dtype=torch.float32, requires_grad=True)
        pred_neg = torch.tensor([[46.0 - d, 46.0, 54.0 - d, 54.0]], dtype=torch.float32, requires_grad=True)

        loss_pos = aligned_h_wiou_loss(pred_pos, target, sigma_0=8.0)
        loss_neg = aligned_h_wiou_loss(pred_neg, target, sigma_0=8.0)

        self.assertAlmostEqual(loss_pos.item(), loss_neg.item(), places=5)

        loss_pos.backward()
        loss_neg.backward()

        grad_pos = pred_pos.grad[0]
        grad_neg = pred_neg.grad[0]

        # Centroid gradient dL/dx_c = dL/dx1 + dL/dx2
        grad_cx_pos = (grad_pos[0] + grad_pos[2]).item()
        grad_cx_neg = (grad_neg[0] + grad_neg[2]).item()

        # Opposite x centroid gradients
        self.assertAlmostEqual(grad_cx_pos + grad_cx_neg, 0.0, places=5)
        self.assertGreater(abs(grad_cx_pos), 1e-4)

    def test_3_exact_box_match_and_local_minimum(self):
        """Property 3: Exact box match yields S=1.0, L=0.0 with finite autograd,
        and loss strictly increases under any perturbation in either direction (local minimum).
        """
        target = torch.tensor([[20.0, 20.0, 30.0, 30.0]], dtype=torch.float64)
        pred = torch.tensor([[20.0, 20.0, 30.0, 30.0]], dtype=torch.float64, requires_grad=True)

        loss = aligned_h_wiou_loss(pred, target, sigma_0=8.0)
        self.assertAlmostEqual(loss.item(), 0.0, places=6)

        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())

        # Test local minimum: any perturbation strictly increases loss
        eps = 1.0
        for dx, dy in [(eps, 0.0), (-eps, 0.0), (0.0, eps), (0.0, -eps)]:
            p_pert = torch.tensor([[20.0 + dx, 20.0 + dy, 30.0 + dx, 30.0 + dy]], dtype=torch.float64)
            l_pert = aligned_h_wiou_loss(p_pert, target, sigma_0=8.0)
            self.assertGreater(l_pert.item(), 0.0, f"Loss did not increase at ({dx}, {dy})")

    def test_4_asymptotic_similarity_limits(self):
        """Property 4: Full similarity converges to IoU at large scale and to exp(-D_SN^2) at micro scale."""
        # 4a. Large target (s_B = 1000 px >> sigma_0 = 8 px)
        # Target [0, 0, 1000, 1000] (center 500, 500, size 1000, 1000)
        # Pred [100, 100, 1000, 1000] (center 550, 550, size 900, 900)
        # Overlap: inter = 900*900 = 810,000; union = 1,000,000 + 810,000 - 810,000 = 1,000,000 => IoU = 0.81
        large_pred = torch.tensor([[100.0, 100.0, 1000.0, 1000.0]], dtype=torch.float64)
        large_target = torch.tensor([[0.0, 0.0, 1000.0, 1000.0]], dtype=torch.float64)

        loss_large = aligned_h_wiou_loss(large_pred, large_target, sigma_0=8.0)
        sim_large = 1.0 - loss_large.item()
        exact_iou = 0.81
        self.assertAlmostEqual(sim_large, exact_iou, places=3, msg="At s_B=1000, S does not match IoU")

        # 4b. Micro target (s_B = 0.5 px << sigma_0 = 8 px)
        # Disjoint target [0, 0, 0.5, 0.5] (center 0.25, 0.25)
        # Disjoint pred [2.0, 2.0, 2.5, 2.5] (center 2.25, 2.25)
        # Exact IoU = 0.0
        micro_pred = torch.tensor([[2.0, 2.0, 2.5, 2.5]], dtype=torch.float64)
        micro_target = torch.tensor([[0.0, 0.0, 0.5, 0.5]], dtype=torch.float64)

        loss_micro = aligned_h_wiou_loss(micro_pred, micro_target, sigma_0=8.0)
        sim_micro = 1.0 - loss_micro.item()

        # Compute exp(-D_SN^2)
        d_sn_sq = aligned_scale_normalized_divergence_squared(
            torch.tensor([2.25], dtype=torch.float64), torch.tensor([2.25], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64), torch.tensor([0.5], dtype=torch.float64),
            torch.tensor([0.25], dtype=torch.float64), torch.tensor([0.25], dtype=torch.float64),
            torch.tensor([0.5], dtype=torch.float64), torch.tensor([0.5], dtype=torch.float64),
        )
        expected_transport_sim = torch.exp(-d_sn_sq).item()
        self.assertAlmostEqual(sim_micro, expected_transport_sim, places=3, msg="At s_B=0.5, S does not match exp(-D_SN^2)")

    def test_5_numerical_gradient_finite_difference_check(self):
        """Property 5: Autograd gradient matches central finite differences in float64."""
        eps = 1e-5
        target = torch.tensor([[10.0, 10.0, 18.0, 18.0]], dtype=torch.float64)
        pred_base = torch.tensor([[15.0, 12.0, 25.0, 22.0]], dtype=torch.float64)

        pred = pred_base.clone().detach().requires_grad_(True)
        loss = aligned_h_wiou_loss(pred, target, sigma_0=8.0)
        loss.backward()
        analytical_grad = pred.grad.clone()

        numerical_grad = torch.zeros_like(pred_base)
        for i in range(4):
            pred_plus = pred_base.clone()
            pred_minus = pred_base.clone()
            pred_plus[0, i] += eps
            pred_minus[0, i] -= eps

            l_plus = aligned_h_wiou_loss(pred_plus, target, sigma_0=8.0)
            l_minus = aligned_h_wiou_loss(pred_minus, target, sigma_0=8.0)
            numerical_grad[0, i] = (l_plus - l_minus) / (2.0 * eps)

        diff = torch.abs(analytical_grad - numerical_grad)
        rel_diff = diff / (torch.abs(analytical_grad) + torch.abs(numerical_grad) + 1e-8)
        self.assertTrue((rel_diff < 1e-4).all(), f"Finite difference mismatch in float64: analytical={analytical_grad}, numerical={numerical_grad}, rel_diff={rel_diff}")

    def test_6_no_nan_inf_degenerate_clamps(self):
        """Property 6: Clamped zero-size or huge coordinates produce no NaN or Inf."""
        degenerate_preds = [
            torch.tensor([[10.0, 10.0, 10.0, 10.0]], dtype=torch.float32),
            torch.tensor([[-100.0, -100.0, 10000.0, 10.0]], dtype=torch.float32),
            torch.tensor([[1e6, 1e6, 1e6 + 5.0, 1e6 + 5.0]], dtype=torch.float32),
        ]
        target = torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float32)

        for p in degenerate_preds:
            p_grad = p.clone().detach().requires_grad_(True)
            loss = aligned_h_wiou_loss(p_grad, target, sigma_0=8.0)
            self.assertTrue(torch.isfinite(loss).all())
            loss.backward()
            self.assertTrue(torch.isfinite(p_grad.grad).all())

    def test_7_vectorized_vs_scalar_consistency(self):
        """Property 7: Batched vectorized similarity & loss match scalar loop to precision."""
        N = 25
        pred_boxes = torch.rand(N, 4) * 100.0
        pred_boxes[:, 2:] += pred_boxes[:, :2] + 2.0

        target_boxes = torch.rand(N, 4) * 100.0
        target_boxes[:, 2:] += target_boxes[:, :2] + 2.0

        batch_loss = aligned_h_wiou_loss(pred_boxes, target_boxes, sigma_0=8.0)

        scalar_losses = []
        for i in range(N):
            l = aligned_h_wiou_loss(pred_boxes[i:i+1], target_boxes[i:i+1], sigma_0=8.0)
            scalar_losses.append(l)
        scalar_loss = torch.cat(scalar_losses, dim=0)

        self.assertTrue(torch.allclose(batch_loss, scalar_loss, atol=1e-6))

    def test_8_fixed_target_compact_domain_separation_positive_lower_bound(self):
        """Property 8 (Theorem 1 for fixed B): For multiple targets B with diverse aspect ratios,
        and for all delta <= ||mu_a - mu_b||/s_B <= kappa in compact domain,
        gradient norm is strictly bounded below by c(B, delta, kappa, r_min, r_max) > 0.
        """
        # Test targets with diverse aspect ratios (square, horizontal, vertical, extreme)
        target_configs = [
            (8.0, 8.0),   # 1:1 square
            (16.0, 4.0),  # 4:1 horizontal
            (32.0, 2.0),  # 16:1 extreme horizontal
            (4.0, 16.0),  # 1:4 vertical
            (2.0, 32.0),  # 1:16 extreme vertical
        ]

        delta = 0.5
        kappa = 2.0
        r_max = 1.5

        for wb, hb in target_configs:
            s_B = math.sqrt(wb * hb)
            target = torch.tensor([[50.0 - wb/2, 50.0 - hb/2, 50.0 + wb/2, 50.0 + hb/2]], dtype=torch.float64)

            # Theoretical lower bound constant for this fixed B:
            # m = 4 * delta * s_B / ((1 + r_max^2) * max(wb^2, hb^2))
            # M = 2 * kappa^2 * s_B^2 / min(wb^2, hb^2) + 2 * ln^2(r_max)
            # c(B) = (1 - gamma(s_B)) * exp(-M) * m > 0
            gamma_val = (s_B**2) / (s_B**2 + 8.0**2)
            M = (2.0 * (kappa**2) * (s_B**2)) / min(wb**2, hb**2) + 2.0 * (math.log(r_max)**2)
            m = (4.0 * delta * s_B) / ((1.0 + r_max**2) * max(wb**2, hb**2))
            c_theoretical = (1.0 - gamma_val) * math.exp(-M) * m
            self.assertGreater(c_theoretical, 0.0)

            grad_norms = []
            for sep_ratio in [0.5, 1.0, 1.5, 2.0]:
                dist = sep_ratio * s_B
                for angle_deg in [0, 45, 90, 135, 180, 225, 270, 315]:
                    rad = angle_deg * 3.14159265 / 180.0
                    dx = dist * math.cos(rad)
                    dy = dist * math.sin(rad)

                    # Strictly disjoint box in interior:
                    wa = wb
                    ha = hb
                    pred = torch.tensor([[50.0 + dx - wa/2, 50.0 + dy - ha/2, 50.0 + dx + wa/2, 50.0 + dy + ha/2]], dtype=torch.float64, requires_grad=True)
                    loss = aligned_h_wiou_loss(pred, target, sigma_0=8.0)
                    loss.backward()

                    grad = pred.grad[0]
                    cx_grad = (grad[0] + grad[2]).item()
                    cy_grad = (grad[1] + grad[3]).item()
                    norm = math.sqrt(cx_grad**2 + cy_grad**2)
                    grad_norms.append(norm)

            min_norm = min(grad_norms)
            self.assertGreater(min_norm, 0.0, f"Observed min norm is not strictly positive for B=({wb}, {hb}): {min_norm}")
            self.assertGreaterEqual(min_norm, c_theoretical * 0.99, f"Gradient norm fell below theoretical lower bound for B=({wb}, {hb}): min_norm={min_norm}, c={c_theoretical}")

    def test_9_homothetic_scaling_asymptotics_proposition_3(self):
        """Property 9 (Proposition 3): Homothetic scaling A_lambda = lambda * A, B_lambda = lambda * B
        preserves exact IoU and exact SNGD divergence D_SN^2, while S(A_lambda, B_lambda)
        smoothly transitions from exp(-D_SN^2) (at lambda -> 0) to IoU (at lambda -> inf).
        """
        # Base boxes A and B
        box_a = torch.tensor([[10.0, 10.0, 30.0, 30.0]], dtype=torch.float64)  # center (20, 20), size (20, 20)
        box_b = torch.tensor([[15.0, 15.0, 35.0, 35.0]], dtype=torch.float64)  # center (25, 25), size (20, 20)

        # Baseline invariants
        from torchvision.ops import box_iou
        base_iou = box_iou(box_a, box_b).item()

        xa, ya, wa, ha = (box_a[:, 0] + box_a[:, 2]) / 2, (box_a[:, 1] + box_a[:, 3]) / 2, box_a[:, 2] - box_a[:, 0], box_a[:, 3] - box_a[:, 1]
        xb, yb, wb, hb = (box_b[:, 0] + box_b[:, 2]) / 2, (box_b[:, 1] + box_b[:, 3]) / 2, box_b[:, 2] - box_b[:, 0], box_b[:, 3] - box_b[:, 1]
        base_d_sn_sq = aligned_scale_normalized_divergence_squared(xa, ya, wa, ha, xb, yb, wb, hb).item()
        base_transport = math.exp(-base_d_sn_sq)

        # Verify geometric invariance under diverse lambda
        for lam in [0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0]:
            a_lam = box_a * lam
            b_lam = box_b * lam
            lam_iou = box_iou(a_lam, b_lam).item()
            self.assertAlmostEqual(lam_iou, base_iou, places=6, msg=f"IoU not invariant under lambda={lam}")

            xa_l, ya_l, wa_l, ha_l = (a_lam[:, 0] + a_lam[:, 2]) / 2, (a_lam[:, 1] + a_lam[:, 3]) / 2, a_lam[:, 2] - a_lam[:, 0], a_lam[:, 3] - a_lam[:, 1]
            xb_l, yb_l, wb_l, hb_l = (b_lam[:, 0] + b_lam[:, 2]) / 2, (b_lam[:, 1] + b_lam[:, 3]) / 2, b_lam[:, 2] - b_lam[:, 0], b_lam[:, 3] - b_lam[:, 1]
            lam_d_sn_sq = aligned_scale_normalized_divergence_squared(xa_l, ya_l, wa_l, ha_l, xb_l, yb_l, wb_l, hb_l).item()
            self.assertAlmostEqual(lam_d_sn_sq, base_d_sn_sq, places=6, msg=f"SNGD not invariant under lambda={lam}")

        # Asymptotic limit lambda -> inf: S -> IoU
        a_inf = box_a * 1000.0
        b_inf = box_b * 1000.0
        s_inf = 1.0 - aligned_h_wiou_loss(a_inf, b_inf, sigma_0=8.0).item()
        self.assertAlmostEqual(s_inf, base_iou, places=4, msg="S does not converge to IoU at large lambda")

        # Asymptotic limit lambda -> 0: S -> exp(-D_SN^2)
        a_zero = box_a * 0.001
        b_zero = box_b * 0.001
        s_zero = 1.0 - aligned_h_wiou_loss(a_zero, b_zero, sigma_0=8.0).item()
        self.assertAlmostEqual(s_zero, base_transport, places=4, msg="S does not converge to exp(-D_SN^2) at small lambda")

    def test_10_proposition_2_corrective_descent_direction(self):
        """Property 10 (Proposition 2): For any disjoint box in the interior int(Z_B),
        < -grad L, mu_a - mu_b > < 0 strictly (descent direction points toward target centroid).
        """
        target = torch.tensor([[50.0, 50.0, 60.0, 60.0]], dtype=torch.float64)  # center (55, 55)
        for angle in range(0, 360, 30):
            rad = angle * math.pi / 180.0
            dist = 30.0  # strictly outside box
            dx = dist * math.cos(rad)
            dy = dist * math.sin(rad)
            pred = torch.tensor([[50.0 + dx, 50.0 + dy, 60.0 + dx, 60.0 + dy]], dtype=torch.float64, requires_grad=True)

            loss = aligned_h_wiou_loss(pred, target, sigma_0=8.0)
            loss.backward()

            grad = pred.grad[0]
            cx_grad = (grad[0] + grad[2]).item()
            cy_grad = (grad[1] + grad[3]).item()

            inner_prod = cx_grad * dx + cy_grad * dy
            self.assertGreater(inner_prod, 0.0, f"Positive inner product failed at angle {angle}")

            descent_inner_prod = -inner_prod
            self.assertLess(descent_inner_prod, 0.0, f"Descent direction failed to point towards target at angle {angle}")

    def test_11_fp16_amp_numerical_stability_in_compact_domain(self):
        """Property 11: In IEEE FP16, loss and gradient within the compact domain K remain strictly finite and non-vanishing."""
        target = torch.tensor([[10.0, 10.0, 18.0, 18.0]], dtype=torch.float16)
        pred = torch.tensor([[25.0, 25.0, 33.0, 33.0]], dtype=torch.float16, requires_grad=True)

        loss = aligned_h_wiou_loss(pred, target, sigma_0=8.0)
        self.assertTrue(torch.isfinite(loss).all())
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertFalse(torch.all(pred.grad == 0.0), "FP16 gradient underflowed to exact zero within compact domain")


if __name__ == '__main__':
    unittest.main()