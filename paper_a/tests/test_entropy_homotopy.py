"""
Unit Tests for Entropy-Modulated Homotopy (EH-WIoU) Metric and Loss.
"""
import math
import unittest
import torch

from common.metrics.entropy_homotopy import (
    compute_shannon_entropy,
    EntropyGuidanceModule,
    compute_entropy_homotopy_similarity,
    EntropyHomotopyLoss,
)


class TestEntropyHomotopy(unittest.TestCase):
    def test_shannon_entropy_bounds_and_invariance(self):
        # Uniform distribution across channels -> Max Entropy = 1.0
        B, C, H, W = 2, 8, 16, 16
        uniform_feat = torch.ones((B, C, H, W)) * 5.0
        entropy = compute_shannon_entropy(uniform_feat)
        self.assertEqual(entropy.shape, (B, 1, H, W))
        self.assertTrue(torch.all(entropy >= 0.0) and torch.all(entropy <= 1.0))
        # For uniform distribution, normalized entropy should be ~1.0
        self.assertTrue(torch.allclose(entropy, torch.ones_like(entropy), atol=1e-3))

        # One-hot distribution -> Min Entropy = 0.0
        one_hot_feat = torch.zeros((B, C, H, W))
        one_hot_feat[:, 0, :, :] = 100.0
        min_entropy = compute_shannon_entropy(one_hot_feat)
        self.assertTrue(torch.all(min_entropy >= 0.0))
        self.assertTrue(torch.all(min_entropy < 0.05))

    def test_entropy_guidance_module_forward_backward(self):
        module = EntropyGuidanceModule(in_channels=256, reduction=4)
        x = torch.randn(2, 256, 32, 32, requires_grad=True)
        enhanced, weights = module(x)
        self.assertEqual(enhanced.shape, (2, 256, 32, 32))
        self.assertEqual(weights.shape, (2, 1, 32, 32))
        self.assertTrue(torch.all(weights >= 0.0) and torch.all(weights <= 1.0))

        loss = enhanced.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_entropy_homotopy_similarity_geometry(self):
        # 2 boxes: completely disjoint (IoU = 0)
        p = torch.tensor([[10.0, 10.0, 14.0, 14.0]])  # 4x4 box
        g = torch.tensor([[20.0, 20.0, 24.0, 24.0]])  # 4x4 box
        sim = compute_entropy_homotopy_similarity(p, g, sigma_0=8.0)
        self.assertEqual(sim.shape, (1, 1))
        # Non-vanishing positive similarity
        self.assertTrue(sim.item() > 0.0)

        # Identical boxes (IoU = 1, W2 = 0) -> Similarity = 1.0
        sim_id = compute_entropy_homotopy_similarity(p, p, sigma_0=8.0)
        self.assertAlmostEqual(sim_id.item(), 1.0, places=4)

    def test_entropy_homotopy_loss_gradient_flow(self):
        loss_fn = EntropyHomotopyLoss(sigma_0=8.0, beta=0.5)
        p = torch.tensor([[10.0, 10.0, 14.0, 14.0]], requires_grad=True)
        g = torch.tensor([[25.0, 25.0, 29.0, 29.0]])  # Disjoint
        h_prior = torch.tensor([0.8])

        loss = loss_fn(p, g, entropy_prior=h_prior)
        self.assertTrue(0.0 <= loss.item() <= 1.0)
        loss.backward()
        self.assertIsNotNone(p.grad)
        self.assertTrue(torch.isfinite(p.grad).all())
        # Non-zero gradient under disjoint alignment
        self.assertTrue(p.grad.abs().sum().item() > 0.0)

    def test_aligned_entropy_homotopy_loss_dual_calling_convention(self):
        from common.metrics.entropy_homotopy import aligned_entropy_homotopy_loss
        from common.metrics import configure_metric

        # 1. 2-Tensor Calling format [N, 4]
        p = torch.tensor([[10.0, 10.0, 18.0, 18.0], [50.0, 50.0, 60.0, 60.0]], requires_grad=True)
        g = torch.tensor([[12.0, 12.0, 20.0, 20.0], [55.0, 55.0, 65.0, 65.0]])
        loss_2t = aligned_entropy_homotopy_loss(p, g, sigma_0=8.0, beta=0.5)
        self.assertEqual(loss_2t.shape, (2,))
        self.assertTrue(torch.all(loss_2t >= 0.0) and torch.all(loss_2t <= 1.0))

        # 2. 8-Coordinate Calling format [N]
        xa = (p[:, 0] + p[:, 2]) / 2.0
        ya = (p[:, 1] + p[:, 3]) / 2.0
        wa = p[:, 2] - p[:, 0]
        ha = p[:, 3] - p[:, 1]

        xb = (g[:, 0] + g[:, 2]) / 2.0
        yb = (g[:, 1] + g[:, 3]) / 2.0
        wb = g[:, 2] - g[:, 0]
        hb = g[:, 3] - g[:, 1]

        loss_8c = aligned_entropy_homotopy_loss(xa, ya, wa, ha, xb, yb, wb, hb, sigma_0=8.0, beta=0.5)
        self.assertEqual(loss_8c.shape, (2,))
        self.assertTrue(torch.allclose(loss_2t, loss_8c, atol=1e-5))

        # 3. Registry & configure_metric contract
        sim_fn, dist_fn, meta = configure_metric("eh_wiou", h_wiou_sigma_0=8.0, h_wiou_static_gamma=0.5)
        self.assertIsNotNone(sim_fn)
        self.assertIsNotNone(dist_fn)
        reg_loss = dist_fn(p, g)
        self.assertTrue(torch.allclose(reg_loss, loss_2t, atol=1e-5))


if __name__ == "__main__":
    unittest.main()

