"""
Unit tests for Cascade Multi-Stage Homotopy Loss & Assigner:
1. Multi-Stage Scale Homotopy schedule validation (sigmas = [8.0, 4.0, 2.0]).
2. Non-vanishing gradient propagation through all cascade stages.
3. Microscopic sample survival test (preventing positive anchor starvation in Stage 2/3).
"""
import unittest
import torch

from common.metrics.cascade_homotopy import (
    CascadeHomotopyLoss,
    cascade_homotopy_stage_matcher,
)
from common.metrics.h_wiou import compute_scale_homotopy


class TestCascadeHomotopy(unittest.TestCase):
    def setUp(self):
        self.sigmas = [8.0, 4.0, 2.0]
        self.loss_fn = CascadeHomotopyLoss(sigmas=self.sigmas, loss_weights=[1.0, 1.0, 1.0])

    def test_homotopy_schedule_asymptotics(self):
        """Test scale-homotopy parameter gamma across multiple stages."""
        micro_wh = torch.tensor([[2.0, 2.0]])
        normal_wh = torch.tensor([[64.0, 64.0]])

        for k, sigma in enumerate(self.sigmas):
            gamma_micro = compute_scale_homotopy(micro_wh, sigma_0=sigma)
            gamma_normal = compute_scale_homotopy(normal_wh, sigma_0=sigma)

            # Micro scale gamma should be <= 0.5 (Wasserstein dominant)
            self.assertTrue(0.0 <= gamma_micro.item() <= 0.5)
            # Normal scale gamma should be close to 1.0 (IoU dominant)
            self.assertTrue(gamma_normal.item() > 0.95)

    def test_multi_stage_loss_gradient_flow(self):
        """Test that gradients backpropagate smoothly across all 3 stages without NaN or vanishing."""
        pred_boxes = torch.tensor([[10.0, 10.0, 14.0, 14.0], [50.0, 50.0, 56.0, 56.0]], requires_grad=True)
        target_boxes = torch.tensor([[11.0, 11.0, 15.0, 15.0], [50.0, 50.0, 56.0, 56.0]])

        for stage_idx in range(len(self.sigmas)):
            loss = self.loss_fn(stage_idx, pred_boxes, target_boxes).sum()
            self.assertFalse(torch.isnan(loss))
            self.assertFalse(torch.isinf(loss))
            self.assertTrue(loss.item() > 0.0)

            if pred_boxes.grad is not None:
                pred_boxes.grad.zero_()
            loss.backward(retain_graph=True)
            self.assertIsNotNone(pred_boxes.grad)
            self.assertTrue(torch.all(torch.isfinite(pred_boxes.grad)))
            self.assertTrue(torch.any(pred_boxes.grad.abs() > 0.0))

    def test_micro_sample_survival_rate(self):
        """Test that micro objects retain positive samples in Stage 2 & 3 via homotopy matcher."""
        gt_boxes = torch.tensor([[100.0, 100.0, 104.0, 104.0]])
        gt_labels = torch.tensor([1])

        # 5 proposals: shifted by 1-3 pixels
        proposals = torch.tensor([
            [101.0, 101.0, 105.0, 105.0],  # 1px shift
            [102.0, 102.0, 106.0, 106.0],  # 2px shift
            [103.0, 103.0, 107.0, 107.0],  # 3px shift
            [200.0, 200.0, 250.0, 250.0],  # far background
            [300.0, 300.0, 350.0, 350.0],  # far background
        ])

        for stage_idx in range(len(self.sigmas)):
            matched_gt, matched_lbl, pos_mask = cascade_homotopy_stage_matcher(
                proposals, gt_boxes, gt_labels,
                stage_idx=stage_idx,
                sigmas=self.sigmas,
                pos_thresh=0.35,
                neg_thresh=0.15,
                topk_fallback=2,
            )
            num_pos = pos_mask.sum().item()
            self.assertTrue(num_pos >= 2, f"Stage {stage_idx} failed: only {num_pos} positive samples survived!")


if __name__ == "__main__":
    unittest.main()
