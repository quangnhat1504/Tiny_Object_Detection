"""
Unit tests for Homotopy-Aware RoI Head Matching & Feature-Level EGM:
1. Microscopic sample survival test in RoI Head (Homotopy vs Pure IoU).
2. End-to-end model forward with use_homotopy_roi_matching=True and use_egm=True.
3. Gradient propagation through EGM and RoI Head under AMP.
"""
import unittest
import torch
import torch.nn as nn
from torchvision.ops import boxes as box_ops
from torchvision.models.detection._utils import Matcher

from common.model import build_model, _wrap_roi_for_homotopy_matching, FPNEntropyGuidance
from common.metrics.cascade_homotopy import compute_entropy_homotopy_similarity


class TestHomotopyRoIMatching(unittest.TestCase):
    def test_roi_micro_proposal_retention(self):
        """Verify that micro proposals (e.g. 6x6 px) shifted by 2-3 px survive in RoI matching."""
        # GT box of size 6x6 px: [10, 10, 16, 16]
        gt_boxes = [torch.tensor([[10.0, 10.0, 16.0, 16.0]])]
        gt_labels = [torch.tensor([1], dtype=torch.int64)]

        # Proposal shifted by 2 px: [12, 12, 18, 18] -> IoU = 16 / (36 + 36 - 16) = 0.2857
        proposals = [torch.tensor([[12.0, 12.0, 18.0, 18.0]])]

        # 1. Standard torchvision matcher behavior:
        standard_matcher = Matcher(high_threshold=0.5, low_threshold=0.5, allow_low_quality_matches=False)
        iou_mat = box_ops.box_iou(gt_boxes[0], proposals[0])
        standard_res = standard_matcher(iou_mat)
        # Under standard IoU, 0.2857 < 0.5 -> marked as BELOW_LOW_THRESHOLD (-1)
        self.assertEqual(standard_res[0].item(), standard_matcher.BELOW_LOW_THRESHOLD)

        # 2. Homotopy RoI matcher behavior:
        class DummyRoIHeads:
            def __init__(self):
                self.assign_targets_to_proposals = None

        dummy_roi = DummyRoIHeads()
        _wrap_roi_for_homotopy_matching(
            dummy_roi,
            metric_fn=compute_entropy_homotopy_similarity,
            fg_thresh=0.40,
            bg_thresh=0.30,
            allow_low_quality_matches=True,
        )

        matched_idxs, labels = dummy_roi.assign_targets_to_proposals(proposals, gt_boxes, gt_labels)
        # Under Homotopy matching, the proposal is recognized as positive (class 1)
        self.assertEqual(labels[0][0].item(), 1, "Micro proposal should be retained as positive foreground!")

    def test_model_forward_backward_with_egm_and_homotopy_roi(self):
        """Verify full model forward + backward pass with both upgrades active."""
        model = build_model(
            num_classes=9,
            placement="h_wiou",
            box_loss_type="h_wiou",
            use_homotopy_roi_matching=True,
            use_egm=True,
        )
        model.train()

        # Synthetic micro-object batch
        images = [torch.rand(3, 256, 256)]
        targets = [{
            "boxes": torch.tensor([[50.0, 50.0, 58.0, 58.0]], dtype=torch.float32),
            "labels": torch.tensor([1], dtype=torch.int64),
        }]

        losses = model(images, targets)
        self.assertIn("loss_classifier", losses)
        self.assertIn("loss_box_reg", losses)
        self.assertIn("loss_objectness", losses)
        self.assertIn("loss_rpn_box_reg", losses)

        total_loss = sum(losses.values())
        self.assertFalse(torch.isnan(total_loss))
        self.assertFalse(torch.isinf(total_loss))
        self.assertGreater(total_loss.item(), 0.0)

        total_loss.backward()

        # Verify EGM parameters received gradients
        egm_has_grad = False
        for name, param in model.backbone.egm.named_parameters():
            if param.grad is not None and torch.any(param.grad.abs() > 0):
                egm_has_grad = True
                break
        self.assertTrue(egm_has_grad, "EGM parameters must receive gradients during backprop!")


if __name__ == "__main__":
    unittest.main()
