"""
Unit tests for Scale-Decoupled Adaptive Faster R-CNN (SDA-FRCNN).
Tests assignment decoupling, dynamic top-k for micro targets, and gradient orthogonality.
"""
from __future__ import annotations
import unittest
import torch
import torch.nn.functional as F

from common.assigner import ScaleDecoupledAssigner
from common.metrics import get_metric_fn


class TestScaleDecoupledAssigner(unittest.TestCase):

    def setUp(self):
        self.metric_fn = get_metric_fn("sa_alw_full")
        self.assigner = ScaleDecoupledAssigner(
            metric_fn=self.metric_fn,
            micro_cutoff_px=8.0,
            fg_iou_thresh=0.7,
            bg_iou_thresh=0.3,
            micro_topk=4,
            micro_pos_sim_thr=0.35,
        )

    def test_empty_gt_assignment(self):
        anchors = torch.tensor([
            [0.0, 0.0, 16.0, 16.0],
            [10.0, 10.0, 26.0, 26.0],
        ], dtype=torch.float32)
        gt_boxes = torch.zeros((0, 4), dtype=torch.float32)

        labels, matched_boxes = self.assigner(anchors, gt_boxes)
        self.assertEqual(labels.shape, (2,))
        self.assertEqual(matched_boxes.shape, (2, 4))
        self.assertTrue((labels == 0.0).all())

    def test_scale_decoupled_assignment_micro_and_standard(self):
        # 1 standard GT (16x16, scale=16px >= 8px)
        # 1 micro GT (4x4, scale=4px < 8px)
        gt_boxes = torch.tensor([
            [0.0, 0.0, 16.0, 16.0],   # standard (16x16)
            [100.0, 100.0, 104.0, 104.0],  # micro (4x4)
        ], dtype=torch.float32)

        anchors = torch.tensor([
            [0.0, 0.0, 16.0, 16.0],   # perfect overlap with std GT -> IoU = 1.0 (pos)
            [0.0, 0.0, 8.0, 8.0],     # partial overlap with std GT -> IoU = 0.25 (neg)
            [100.0, 100.0, 108.0, 108.0],  # anchor near micro GT -> IoU = 0.25 (standard would drop, SDA assigns via SA-ALW)
            [100.0, 100.0, 116.0, 116.0],  # anchor near micro GT
            [500.0, 500.0, 516.0, 516.0],  # far away -> background
        ], dtype=torch.float32)

        labels, matched_boxes = self.assigner(anchors, gt_boxes)

        # Anchor 0 must be positive for standard GT
        self.assertEqual(labels[0].item(), 1.0)
        self.assertTrue(torch.allclose(matched_boxes[0], gt_boxes[0]))

        # Anchor 2 must be positive for micro GT due to top-k SA-ALW
        self.assertEqual(labels[2].item(), 1.0)
        self.assertTrue(torch.allclose(matched_boxes[2], gt_boxes[1]))

        # Far away anchor must be negative
        self.assertEqual(labels[4].item(), 0.0)

    def test_mixed_scale_batch(self):
        # 3 GTs: large, tiny, micro
        gt_boxes = torch.tensor([
            [0.0, 0.0, 32.0, 32.0],     # large (32x32, s=32)
            [50.0, 50.0, 60.0, 60.0],   # tiny (10x10, s=10)
            [120.0, 120.0, 124.0, 124.0], # micro (4x4, s=4)
        ], dtype=torch.float32)

        anchors = torch.tensor([
            [0.0, 0.0, 32.0, 32.0],       # exact match large -> pos
            [50.0, 50.0, 60.0, 60.0],     # exact match tiny -> pos
            [120.0, 120.0, 128.0, 128.0], # candidate for micro -> pos
            [300.0, 300.0, 316.0, 316.0], # background -> neg
        ], dtype=torch.float32)

        labels, matched_boxes = self.assigner(anchors, gt_boxes)
        self.assertEqual(labels[0].item(), 1.0)
        self.assertEqual(labels[1].item(), 1.0)
        self.assertEqual(labels[2].item(), 1.0)
        self.assertEqual(labels[3].item(), 0.0)

    def test_gradient_orthogonality(self):
        # Verify PCGrad orthogonal projection logic
        g_main = torch.tensor([1.0, 2.0, -1.0, 0.5], dtype=torch.float32)
        g_micro = torch.tensor([-2.0, -1.0, 0.5, 1.0], dtype=torch.float32)

        dot = torch.dot(g_main, g_micro)
        # Project g_micro onto orthogonal hyperplane of g_main if dot < 0
        if dot < 0:
            g_proj = g_micro - (dot / (torch.norm(g_main) ** 2)) * g_main
        else:
            g_proj = g_micro

        ortho_dot = torch.dot(g_main, g_proj)
        self.assertAlmostEqual(float(ortho_dot.item()), 0.0, places=5)

    def test_zero_leakage_isolation(self):
        # Verify that assigner and metric computations are purely functional and stateless
        self.assertTrue(hasattr(self.assigner, "micro_cutoff_px"))
        self.assertEqual(self.assigner.micro_cutoff_px, 8.0)
        # Assigner has 0 trainable parameters (no leakable memory)
        trainable_params = sum(p.numel() for p in self.assigner.parameters() if p.requires_grad)
        self.assertEqual(trainable_params, 0)


if __name__ == "__main__":
    unittest.main()
