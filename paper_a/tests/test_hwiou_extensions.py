"""
Unit tests for the 3 Novel H-WIoU v2 Extensions:
1. Dynamic Uncertainty Homotopy (DU-HWIoU)
2. Spectral Wavelet Homotopy (SW-HWIoU)
3. Oriented 2D Gaussian Homotopy (O-HWIoU)
"""
import unittest
import torch

from common.metrics.dynamic_uncertainty_h_wiou import (
    UncertaintyHomotopyPredictor,
    compute_dynamic_scale_homotopy,
    aligned_dynamic_uncertainty_h_wiou_loss,
)
from common.metrics.wavelet_h_wiou import (
    haar_wavelet_2d,
    compute_spectral_high_frequency_ratio,
    compute_wavelet_spectral_homotopy,
    aligned_wavelet_spectral_h_wiou_loss,
)
from common.metrics.oriented_h_wiou import (
    oriented_box_to_2d_gaussian,
    oriented_wasserstein_distance_squared,
    oriented_h_wiou_similarity,
)


class TestHWIoUExtensions(unittest.TestCase):
    def test_dynamic_uncertainty_module(self):
        predictor = UncertaintyHomotopyPredictor(in_features=256, hidden_dim=64, sigma_base=8.0)
        dummy_feat = torch.randn(10, 256)
        sigma_0 = predictor(dummy_feat)
        self.assertEqual(sigma_0.shape, (10,))
        self.assertTrue(torch.all(sigma_0 >= 1.0))

        pred_b = torch.tensor([[10.0, 10.0, 20.0, 20.0], [50.0, 50.0, 56.0, 56.0]])
        tgt_b = torch.tensor([[11.0, 11.0, 21.0, 21.0], [50.0, 50.0, 56.0, 56.0]])
        loss = aligned_dynamic_uncertainty_h_wiou_loss(pred_b, tgt_b, sigma_0=sigma_0[:2], sigma_base=8.0)
        self.assertEqual(loss.shape, (2,))
        self.assertTrue(torch.all(loss >= 0.0))

    def test_spectral_wavelet_module(self):
        dummy_feat = torch.randn(2, 64, 32, 32)
        ll, lh, hl, hh = haar_wavelet_2d(dummy_feat)
        self.assertEqual(ll.shape, (2, 64, 16, 16))
        self.assertEqual(lh.shape, (2, 64, 16, 16))

        rho = compute_spectral_high_frequency_ratio(dummy_feat)
        self.assertEqual(rho.shape, (2,))
        self.assertTrue(torch.all(rho >= 0.0))

        pred_b = torch.tensor([[10.0, 10.0, 18.0, 18.0]])
        tgt_b = torch.tensor([[10.0, 10.0, 18.0, 18.0]])
        loss = aligned_wavelet_spectral_h_wiou_loss(pred_b, tgt_b, sigma_0=8.0, spectral_rho=rho[0])
        self.assertAlmostEqual(loss.item(), 0.0, places=4)

    def test_oriented_homotopy_module(self):
        x1 = torch.tensor([100.0])
        y1 = torch.tensor([100.0])
        w1 = torch.tensor([20.0])
        h1 = torch.tensor([10.0])
        theta1 = torch.tensor([0.0])

        # Exact match
        sim_identical = oriented_h_wiou_similarity(
            x1, y1, w1, h1, theta1, x1, y1, w1, h1, theta1, sigma_0=8.0
        )
        self.assertAlmostEqual(sim_identical.item(), 1.0, places=3)

        # Rotated 90 degrees
        theta2 = torch.tensor([1.5708])
        sim_rotated = oriented_h_wiou_similarity(
            x1, y1, w1, h1, theta1, x1, y1, w1, h1, theta2, sigma_0=8.0
        )
        self.assertTrue(0.0 <= sim_rotated.item() < 1.0)


if __name__ == "__main__":
    unittest.main()
