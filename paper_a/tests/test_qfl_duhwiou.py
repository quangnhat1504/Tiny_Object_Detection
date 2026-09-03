"""
Unit tests for Task-Aligned Quality Focal Loss (QFL) + H-WIoU and Dynamic Uncertainty Homotopy (DU-HWIoU).
"""
import unittest
import torch

from common.metrics import configure_metric
from common.metrics.dynamic_uncertainty_h_wiou import (
    UncertaintyHomotopyPredictor,
    aligned_dynamic_uncertainty_h_wiou_loss,
)
from common.model import build_model


class TestQFLAndDUHWIoU(unittest.TestCase):
    def test_qfl_with_hwiou_forward_backward(self):
        """Test Faster R-CNN with QFL classification and H-WIoU regression."""
        sim_fn, dist_fn, _ = configure_metric("h_wiou", h_wiou_sigma_0=8.0)
        model = build_model(
            num_classes=9,
            metric_fn=sim_fn,
            metric_distance_fn=dist_fn,
            placement="la_loss",
            box_loss_type="h_wiou",
            use_quality_focal=True,
            quality_focal_beta=2.0,
        )
        model.train()

        imgs = [torch.randn(3, 200, 200), torch.randn(3, 200, 200)]
        targets = [
            {
                "boxes": torch.tensor([[20.0, 20.0, 30.0, 30.0], [50.0, 50.0, 60.0, 60.0]]),
                "labels": torch.tensor([1, 2], dtype=torch.int64),
            },
            {
                "boxes": torch.tensor([[10.0, 10.0, 18.0, 18.0]]),
                "labels": torch.tensor([3], dtype=torch.int64),
            },
        ]

        loss_dict = model(imgs, targets)
        self.assertIn("loss_classifier", loss_dict)
        self.assertIn("loss_box_reg", loss_dict)
        total_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))

        self.assertFalse(torch.isnan(total_loss))
        self.assertFalse(torch.isinf(total_loss))
        self.assertTrue(total_loss.item() > 0.0)

        total_loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                self.assertFalse(torch.isnan(p.grad).any(), f"NaN in gradient of {name}")

    def test_dynamic_uncertainty_homotopy_loss(self):
        """Test DU-HWIoU loss computation with adaptive instance scale."""
        pred_b = torch.tensor([[10.0, 10.0, 16.0, 16.0], [50.0, 50.0, 56.0, 56.0]], requires_grad=True)
        tgt_b = torch.tensor([[11.0, 11.0, 17.0, 17.0], [50.0, 50.0, 56.0, 56.0]])
        sigma_0 = torch.tensor([6.5, 9.2])

        loss = aligned_dynamic_uncertainty_h_wiou_loss(pred_b, tgt_b, sigma_0=sigma_0, sigma_base=8.0)
        self.assertEqual(loss.shape, (2,))
        self.assertTrue(torch.all(loss >= 0.0))

        total_loss = loss.sum()
        total_loss.backward()
        self.assertIsNotNone(pred_b.grad)
        self.assertTrue(torch.all(torch.isfinite(pred_b.grad)))


if __name__ == "__main__":
    unittest.main()
