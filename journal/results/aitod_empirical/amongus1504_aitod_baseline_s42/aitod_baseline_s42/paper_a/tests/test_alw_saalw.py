from __future__ import annotations

import math
import unittest

import torch
from torchvision.models.detection._utils import BoxCoder

from common.metrics import configure_metric, get_metric_distance_fn
from common.metrics import sa_alw_canonical as canonical
from common.model import _hierarchical_assignment, _metric_aux_loss
from scripts.train_frcnn_metric import _validate_paper_a_protocol


SCHEDULE = {
    "s_min": 5.0,
    "s_max": 20.0,
    "beta_min": 8.0,
    "beta_max": 10.0,
    "w_min": 1.0,
    "w_max": 1.5,
}


def columns(boxes: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(boxes[:, index] for index in range(4))


def aligned(fn, predictions: torch.Tensor, targets: torch.Tensor, **kwargs):
    return fn(*columns(predictions), *columns(targets), **kwargs)


class CanonicalGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.boxes = torch.tensor(
            [[5.0, 7.0, 4.0, 8.0], [13.0, 2.0, 12.0, 3.0]],
            dtype=torch.float64,
        )

    def test_identity_and_non_negativity(self) -> None:
        distance_squared = aligned(
            canonical.aligned_alw_distance_squared, self.boxes, self.boxes
        )
        distance = aligned(canonical.aligned_alw_distance, self.boxes, self.boxes)
        self.assertTrue(torch.equal(distance_squared, torch.zeros_like(distance_squared)))
        self.assertTrue(torch.equal(distance, torch.zeros_like(distance)))

        shifted = self.boxes.clone()
        shifted[:, 0] += torch.tensor([1.0, -2.0], dtype=shifted.dtype)
        result = aligned(canonical.aligned_alw_distance_squared, shifted, self.boxes)
        self.assertTrue(bool((result >= 0).all()))

    def test_squared_log_ratio_is_exact(self) -> None:
        prediction = torch.tensor([[0.0, 0.0, 8.0, 4.0]], dtype=torch.float64)
        target = torch.tensor([[0.0, 0.0, 4.0, 4.0]], dtype=torch.float64)
        result = aligned(
            canonical.aligned_alw_distance_squared, prediction, target
        ).item()
        self.assertAlmostEqual(result, math.log(2.0) ** 2, places=12)

    def test_alw_symmetry(self) -> None:
        other = torch.tensor(
            [[6.5, 3.0, 7.0, 4.0], [9.0, 8.0, 5.0, 11.0]],
            dtype=torch.float64,
        )
        forward = aligned(
            canonical.aligned_alw_distance_squared, self.boxes, other
        )
        reverse = aligned(
            canonical.aligned_alw_distance_squared, other, self.boxes
        )
        torch.testing.assert_close(forward, reverse, rtol=0, atol=1e-12)

    def test_alw_joint_scale_invariance(self) -> None:
        target = torch.tensor(
            [[6.0, 8.0, 6.0, 7.0], [11.0, 3.0, 8.0, 6.0]],
            dtype=torch.float64,
        )
        base = aligned(canonical.aligned_alw_distance_squared, self.boxes, target)
        for factor in (0.25, 3.0, 17.0):
            scaled = aligned(
                canonical.aligned_alw_distance_squared,
                self.boxes * factor,
                target * factor,
            )
            torch.testing.assert_close(base, scaled, rtol=1e-12, atol=1e-12)

    def test_sa_alw_is_intentionally_target_conditioned(self) -> None:
        prediction = torch.tensor([[2.0, 0.0, 6.0, 6.0]], dtype=torch.float64)
        target = torch.tensor([[0.0, 0.0, 18.0, 18.0]], dtype=torch.float64)
        forward = aligned(
            canonical.aligned_sa_alw_distance_squared,
            prediction,
            target,
            **{key: SCHEDULE[key] for key in ("s_min", "s_max", "w_min", "w_max")},
        )
        reverse = aligned(
            canonical.aligned_sa_alw_distance_squared,
            target,
            prediction,
            **{key: SCHEDULE[key] for key in ("s_min", "s_max", "w_min", "w_max")},
        )
        self.assertFalse(torch.allclose(forward, reverse))

    def test_schedule_clips_below_and_above_train_bounds(self) -> None:
        gt_wh = torch.tensor([[1.0, 1.0], [12.5, 12.5], [100.0, 100.0]])
        u = canonical.scale_interpolation(gt_wh, s_min=5.0, s_max=20.0)
        torch.testing.assert_close(u, torch.tensor([1.0, 0.5, 0.0]))
        beta = canonical.scale_adaptive_beta(
            gt_wh,
            s_min=5.0,
            s_max=20.0,
            beta_min=8.0,
            beta_max=10.0,
        )
        weight = canonical.scale_adaptive_position_weight(
            gt_wh,
            s_min=5.0,
            s_max=20.0,
            w_min=1.0,
            w_max=1.5,
        )
        torch.testing.assert_close(beta, torch.tensor([10.0, 9.0, 8.0]))
        torch.testing.assert_close(weight, torch.tensor([1.5, 1.25, 1.0]))

    def test_log_linear_schedule_has_geometric_midpoint(self) -> None:
        geometric_midpoint = math.sqrt(5.0 * 20.0)
        gt_wh = torch.tensor(
            [[2.0, 2.0], [geometric_midpoint, geometric_midpoint], [40.0, 40.0]],
            dtype=torch.float64,
        )
        interpolation = canonical.scale_interpolation(
            gt_wh, s_min=5.0, s_max=20.0, schedule_form="log_linear"
        )
        torch.testing.assert_close(
            interpolation,
            torch.tensor([1.0, 0.5, 0.0], dtype=torch.float64),
            rtol=0,
            atol=1e-12,
        )

    def test_unknown_schedule_form_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "schedule_form"):
            canonical.scale_interpolation(
                torch.ones((1, 2)),
                s_min=5.0,
                s_max=20.0,
                schedule_form="sigmoid",
            )

    def test_tiny_and_near_perfect_gradients_are_finite(self) -> None:
        prediction = torch.tensor(
            [[0.001, -0.001, 0.0101, 0.0199]],
            dtype=torch.float64,
            requires_grad=True,
        )
        target = torch.tensor([[0.0, 0.0, 0.01, 0.02]], dtype=torch.float64)
        loss = aligned(canonical.aligned_alw_distance, prediction, target).sum()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cpu_gpu_and_amp_consistency(self) -> None:
        prediction = self.boxes.float()
        target = (self.boxes * 1.03).float()
        cpu = aligned(canonical.aligned_alw_distance, prediction, target)
        prediction_cuda = prediction.cuda()
        target_cuda = target.cuda()
        gpu = aligned(
            canonical.aligned_alw_distance, prediction_cuda, target_cuda
        ).cpu()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            amp = aligned(
                canonical.aligned_alw_distance, prediction_cuda, target_cuda
            ).float().cpu()
        torch.testing.assert_close(cpu, gpu, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(gpu, amp, rtol=2e-3, atol=2e-4)


class PlacementAndProtocolTests(unittest.TestCase):
    def test_canonical_regression_is_aligned_not_all_pairs(self) -> None:
        prediction_boxes = torch.tensor(
            [[0.0, 0.0, 4.0, 4.0], [20.0, 20.0, 28.0, 28.0]]
        )
        target_boxes = torch.tensor(
            [[1.0, 0.0, 5.0, 4.0], [21.0, 20.0, 29.0, 28.0]]
        )
        similarity, distance, _ = configure_metric("alw_canonical", beta=8.0)
        loss = _metric_aux_loss(
            prediction_boxes,
            target_boxes,
            similarity,
            reliability_thr=16.0,
            metric_distance_fn=distance,
        )
        pred_geometry = torch.stack(
            (
                (prediction_boxes[:, 0] + prediction_boxes[:, 2]) / 2,
                (prediction_boxes[:, 1] + prediction_boxes[:, 3]) / 2,
                prediction_boxes[:, 2] - prediction_boxes[:, 0],
                prediction_boxes[:, 3] - prediction_boxes[:, 1],
            ),
            dim=1,
        )
        target_geometry = torch.stack(
            (
                (target_boxes[:, 0] + target_boxes[:, 2]) / 2,
                (target_boxes[:, 1] + target_boxes[:, 3]) / 2,
                target_boxes[:, 2] - target_boxes[:, 0],
                target_boxes[:, 3] - target_boxes[:, 1],
            ),
            dim=1,
        )
        expected = aligned(
            canonical.aligned_alw_distance, pred_geometry, target_geometry
        ).mean()
        torch.testing.assert_close(loss, expected)

    def test_beta_only_does_not_change_regression_distance(self) -> None:
        _, alw_distance, _ = configure_metric("alw_canonical", beta=8.0)
        _, beta_only_distance, _ = configure_metric(
            "sa_alw_canonical_beta_only", **SCHEDULE
        )
        self.assertIs(beta_only_distance, alw_distance)

    def test_canonical_schedule_must_be_explicit(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit frozen train-derived"):
            configure_metric("sa_alw_canonical")

    def test_configure_metric_records_log_linear_form(self) -> None:
        _, _, metadata = configure_metric(
            "sa_alw_canonical", **SCHEDULE, schedule_form="log_linear"
        )
        self.assertEqual(metadata["schedule_form"], "log_linear")

    def test_paper_protocol_rejects_ap50_selector_and_extra_components(self) -> None:
        with self.assertRaisesRegex(ValueError, "validation COCO AP"):
            _validate_paper_a_protocol(
                metric="sa_alw_canonical",
                placement="la_loss",
                box_loss="metric",
                checkpoint_selector="map50",
                disallowed_components={},
            )
        with self.assertRaisesRegex(ValueError, "out-of-scope"):
            _validate_paper_a_protocol(
                metric="sa_alw_canonical",
                placement="la_loss",
                box_loss="metric",
                checkpoint_selector="coco_ap",
                disallowed_components={"cbl": True},
            )

    def test_hierarchical_conflict_has_one_deterministic_owner(self) -> None:
        similarity = torch.zeros((8, 2))
        similarity[:3, 0] = torch.tensor([1.0, 0.95, 0.9])
        similarity[3:6, 1] = torch.tensor([1.0, 0.95, 0.9])
        similarity[6] = 0.92
        zeros = torch.zeros(8)
        ones = torch.ones(8)
        gt_zeros = torch.zeros(2)
        gt_sizes = torch.full((2,), 10.0)
        matched = _hierarchical_assignment(
            similarity,
            zeros,
            zeros,
            ones,
            ones,
            gt_zeros,
            gt_zeros,
            gt_sizes,
            gt_sizes,
            metric_fn=None,
        )
        self.assertEqual(int(matched[6]), 0)
        self.assertEqual(matched.ndim, 1)

    def test_box_coder_round_trip(self) -> None:
        proposals = torch.tensor(
            [[0.0, 0.0, 8.0, 8.0], [10.0, 12.0, 18.0, 22.0]]
        )
        targets = torch.tensor(
            [[1.0, -1.0, 9.0, 9.0], [9.0, 13.0, 20.0, 24.0]]
        )
        coder = BoxCoder(weights=(1.0, 1.0, 1.0, 1.0))
        encoded = coder.encode_single(targets, proposals)
        decoded = coder.decode_single(encoded, proposals)
        torch.testing.assert_close(decoded, targets, rtol=1e-5, atol=1e-5)

    def test_legacy_metrics_keep_legacy_regression_path(self) -> None:
        self.assertIsNone(get_metric_distance_fn("sa_alw_full"))


if __name__ == "__main__":
    unittest.main()
