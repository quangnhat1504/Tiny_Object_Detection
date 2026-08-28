"""
Unit tests for Homotopy Wasserstein-IoU (H-WIoU) metric and regression loss.
"""
from __future__ import annotations
import unittest
import torch

from common.metrics.h_wiou import (
    compute_scale_homotopy,
    compute_h_wiou_similarity,
    aligned_h_wiou_loss,
)
from common.metrics import get_metric_fn, get_metric_distance_fn
from common.model import build_model


class TestHWIoUMetric(unittest.TestCase):
    def test_scale_homotopy_asymptotics(self):
        # Microscopic scale (s = 2 px) vs sigma_0 = 8.0 px:
        # gamma(2) = 4 / (4 + 64) = 4/68 ~ 0.0588
        micro_wh = torch.tensor([[2.0, 2.0]])
        gamma_micro = compute_scale_homotopy(micro_wh, sigma_0=8.0)
        self.assertLess(gamma_micro.item(), 0.1)

        # Standard scale (s = 64 px) vs sigma_0 = 8.0 px:
        # gamma(64) = 4096 / (4096 + 64) = 4096/4160 ~ 0.9846
        large_wh = torch.tensor([[64.0, 64.0]])
        gamma_large = compute_scale_homotopy(large_wh, sigma_0=8.0)
        self.assertGreater(gamma_large.item(), 0.98)

    def test_similarity_identity_and_bounds(self):
        # Identity boxes
        x = torch.tensor([50.0])
        y = torch.tensor([50.0])
        w = torch.tensor([10.0])
        h = torch.tensor([10.0])

        sim = compute_h_wiou_similarity(x, y, w, h, x, y, w, h, sigma_0=8.0)
        self.assertAlmostEqual(sim.item(), 1.0, places=4)

        # Non-overlapping boxes
        x2 = torch.tensor([200.0])
        sim_disjoint = compute_h_wiou_similarity(x, y, w, h, x2, y, w, h, sigma_0=8.0)
        # Even with IoU = 0, micro/tiny should have smooth positive similarity
        self.assertGreaterEqual(sim_disjoint.item(), 0.0)
        self.assertLess(sim_disjoint.item(), 0.05)

    def test_loss_identity_and_gradients(self):
        pred_boxes = torch.tensor([[10.0, 10.0, 30.0, 30.0]], requires_grad=True)
        target_boxes = torch.tensor([[10.0, 10.0, 30.0, 30.0]])

        # Identity loss must be 0
        loss_ident = aligned_h_wiou_loss(pred_boxes, target_boxes, sigma_0=8.0)
        self.assertAlmostEqual(loss_ident.item(), 0.0, places=4)

        # Offset loss with backward pass
        pred_offset = torch.tensor([[12.0, 12.0, 32.0, 32.0]], requires_grad=True)
        loss = aligned_h_wiou_loss(pred_offset, target_boxes, sigma_0=8.0)
        self.assertGreater(loss.item(), 0.0)

        loss.backward()
        self.assertIsNotNone(pred_offset.grad)
        self.assertTrue(torch.isfinite(pred_offset.grad).all())

    def test_aligned_h_wiou_loss_8_args(self):
        # 8-argument coordinate calling convention
        xa = torch.tensor([20.0, 50.0])
        ya = torch.tensor([20.0, 50.0])
        wa = torch.tensor([20.0, 40.0])
        ha = torch.tensor([20.0, 40.0])
        xb = torch.tensor([22.0, 52.0])
        yb = torch.tensor([22.0, 52.0])
        wb = torch.tensor([20.0, 40.0])
        hb = torch.tensor([20.0, 40.0])

        loss_8 = aligned_h_wiou_loss(xa, ya, wa, ha, xb, yb, wb, hb, sigma_0=8.0)
        self.assertEqual(loss_8.shape, (2,))
        self.assertTrue((loss_8 > 0.0).all())

    def test_registry_integration(self):
        metric_fn = get_metric_fn("h_wiou")
        self.assertIsNotNone(metric_fn)
        dist_fn = get_metric_distance_fn("h_wiou")
        self.assertIsNotNone(dist_fn)

    def test_ablation_homotopy_forms(self):
        wh = torch.tensor([[4.0, 4.0], [16.0, 16.0], [64.0, 64.0]])
        
        # Pure W2
        g_w2 = compute_scale_homotopy(wh, form="pure_w2")
        self.assertTrue((g_w2 == 0.0).all())
        
        # Pure IoU
        g_iou = compute_scale_homotopy(wh, form="pure_iou")
        self.assertTrue((g_iou == 1.0).all())
        
        # Static half
        g_static = compute_scale_homotopy(wh, form="static", static_gamma=0.5)
        self.assertTrue((g_static == 0.5).all())
        
        # Exponential and Sigmoid monotonicity
        g_exp = compute_scale_homotopy(wh, sigma_0=8.0, form="exponential")
        g_sig = compute_scale_homotopy(wh, sigma_0=8.0, form="sigmoid")
        self.assertTrue((g_exp[0] < g_exp[1] < g_exp[2]))
        self.assertTrue((g_sig[0] < g_sig[1] < g_sig[2]))

    def test_model_build_with_h_wiou(self):
        metric_fn = get_metric_fn("h_wiou")
        model = build_model(
            metric_fn=metric_fn,
            placement="h_wiou",
            box_loss_type="h_wiou",
        )
        self.assertIsNotNone(model)

        # Test dummy forward pass
        images = [torch.rand(3, 100, 100)]
        targets = [{
            "boxes": torch.tensor([[10.0, 10.0, 20.0, 20.0], [50.0, 50.0, 56.0, 56.0]]),
            "labels": torch.tensor([1, 1], dtype=torch.int64),
        }]
        losses = model(images, targets)
        self.assertIn("loss_box_reg", losses)
        self.assertTrue(torch.isfinite(losses["loss_box_reg"]))

    def test_chunked_box_iou_equivalence(self):
        from common.model import chunked_box_iou
        from torchvision.ops import boxes as box_ops
        
        # Test N=10 boxes vs M=5000 boxes
        boxes1 = torch.rand(10, 4) * 500
        boxes1[:, 2:] += boxes1[:, :2] + 5.0
        
        boxes2 = torch.rand(5000, 4) * 500
        boxes2[:, 2:] += boxes2[:, :2] + 5.0
        
        iou_std = box_ops.box_iou(boxes1, boxes2)
        iou_chunked = chunked_box_iou(boxes1, boxes2, chunk_size=512)
        
        self.assertEqual(iou_std.shape, iou_chunked.shape)
        self.assertTrue(torch.allclose(iou_std, iou_chunked, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
