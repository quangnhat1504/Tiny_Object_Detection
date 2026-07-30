"""Model builder — metric-aware Faster R-CNN.

Supports 5 placements:
  - "everywhere"        : baseline, standard torchvision, no metric
  - "la"                : metric in RPN label assignment only
  - "la_loss"           : metric in RPN LA + RoIHeads box loss
  - "la_loss_nms"       : metric in RPN LA + RoIHeads box loss + NMS
  - "saalw_assigner"    : SAALWAssigner (threshold-based) in RPN + box loss

Metric loss implementation: we override torchvision's RoIHeads.compute_loss
to replace Smooth-L1 with metric-distance loss.
"""
from __future__ import annotations

import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn)
from torchvision.models.detection.faster_rcnn import (
    FastRCNNPredictor, RoIHeads)
from torchvision.models.detection.rpn import (
    AnchorGenerator, RegionProposalNetwork, RPNHead)
from torchvision.models.resnet import Bottleneck
from torchvision.ops import (
    batched_nms, box_iou, boxes as box_ops,
    complete_box_iou_loss, distance_box_iou_loss,
)

from .config import (
    NUM_CLASSES, MIN_SIZE, MAX_SIZE, BOX_DETECTIONS_PER_IMG,
    NMS_THRESH_TEST, SCORE_THRESH_TRAIN,
    RPN_NUM_PROPOSALS_TRAIN, RPN_NUM_PROPOSALS_TEST, RPN_NMS_THRESH,
    RPN_FG_IOU, RPN_BG_IOU,
    ROI_FG_IOU_THRESH, ROI_BG_IOU_THRESH,
    RFLA_K, RFLA_BETA,
    RFLA_DYNAMIC_K_MICRO, RFLA_DYNAMIC_K_TINY,
    RFLA_DYNAMIC_K_SMALL, RFLA_DYNAMIC_K_LARGE,
    RFLA_QUALITY_RATIO, RFLA_MIN_SIM,
    NMS_METRIC_THRESH,
    BOX_LOSS_TYPE, BOX_LOSS_METRIC_WEIGHT, BOX_LOSS_WARMUP_EPOCHS,
    CBL_ALPHA, CBL_NUM_BINS, CBL_GRID_BETA, CBL_UM_WEIGHT,
)

EPS = 1e-6


class QualityFastRCNNPredictor(nn.Module):
    """Fast R-CNN predictor with an auxiliary localization-quality logit."""

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
        self.quality_score = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
            )
        x = x.flatten(start_dim=1)
        return self.cls_score(x), self.bbox_pred(x), self.quality_score(x)


def _make_cbl_grid(alpha: float, num_bins: int, beta: float) -> torch.Tensor:
    """Build the symmetric interval-nonuniform grid used by C-BBL."""
    if alpha <= 0:
        raise ValueError("CBL alpha must be positive")
    if num_bins < 3:
        raise ValueError("CBL num_bins must be at least 3")
    uniform = torch.linspace(-alpha, alpha, num_bins, dtype=torch.float32)
    if beta <= 0:
        return uniform
    denom = torch.expm1(torch.tensor(alpha * beta, dtype=torch.float32))
    magnitude = alpha * torch.expm1(beta * uniform.abs()) / denom
    return uniform.sign() * magnitude


class CBLFastRCNNPredictor(nn.Module):
    """Fast R-CNN predictor with distributional RoI box localization."""

    is_distributional = True

    def __init__(self, in_channels: int, num_classes: int, *,
                 alpha: float, num_bins: int, grid_beta: float):
        super().__init__()
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_dist = nn.Linear(
            in_channels, num_classes * 4 * num_bins)
        self.register_buffer(
            "cbl_grid",
            _make_cbl_grid(alpha, num_bins, grid_beta),
        )

    def forward(self, x):
        if x.dim() == 4:
            torch._assert(
                list(x.shape[2:]) == [1, 1],
                "CBL predictor expects pooled spatial dimensions [1, 1]",
            )
        x = x.flatten(start_dim=1)
        class_logits = self.cls_score(x)
        dist_logits = self.bbox_dist(x).reshape(
            x.shape[0], self.num_classes, 4, self.num_bins)
        probabilities = F.softmax(dist_logits.float(), dim=-1)
        box_deltas = (
            probabilities * self.cbl_grid.float().view(1, 1, 1, -1)
        ).sum(dim=-1).to(dtype=dist_logits.dtype)
        return class_logits, box_deltas.flatten(start_dim=1), dist_logits


class _DoubleHeadResidualBlock(nn.Module):
    """Project pooled RoI features to the Double-Head regression width."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.identity = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False)
        self.identity_bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.identity_bn(self.identity(x))
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + identity)


class DoubleHeadCBLPredictor(nn.Module):
    """FC classification plus a convolutional distributional box head."""

    is_distributional = True
    is_double_head = True

    def __init__(self, cls_in_channels: int, reg_in_channels: int,
                 num_classes: int, *, alpha: float, num_bins: int,
                 grid_beta: float, num_convs: int = 4,
                 reg_out_channels: int = 1024):
        super().__init__()
        if num_convs < 1:
            raise ValueError("Double-Head num_convs must be positive")
        if reg_out_channels % 4 != 0:
            raise ValueError("Double-Head reg_out_channels must be divisible by 4")

        self.num_classes = num_classes
        self.num_bins = num_bins
        self.cls_score = nn.Linear(cls_in_channels, num_classes)
        self.reg_projection = _DoubleHeadResidualBlock(
            reg_in_channels, reg_out_channels)
        self.reg_convs = nn.ModuleList([
            Bottleneck(
                inplanes=reg_out_channels,
                planes=reg_out_channels // 4,
                norm_layer=nn.BatchNorm2d,
            )
            for _ in range(num_convs)
        ])
        self.reg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.bbox_dist = nn.Linear(
            reg_out_channels, num_classes * 4 * num_bins)
        self.register_buffer(
            "cbl_grid",
            _make_cbl_grid(alpha, num_bins, grid_beta),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.zeros_(self.cls_score.bias)
        nn.init.normal_(self.bbox_dist.weight, std=0.001)
        nn.init.zeros_(self.bbox_dist.bias)

    def forward(self, cls_features, reg_features):
        cls_features = cls_features.flatten(start_dim=1)
        class_logits = self.cls_score(cls_features)

        reg_features = self.reg_projection(reg_features)
        for block in self.reg_convs:
            reg_features = block(reg_features)
        reg_features = self.reg_pool(reg_features).flatten(start_dim=1)
        dist_logits = self.bbox_dist(reg_features).reshape(
            reg_features.shape[0], self.num_classes, 4, self.num_bins)
        probabilities = F.softmax(dist_logits.float(), dim=-1)
        box_deltas = (
            probabilities * self.cbl_grid.float().view(1, 1, 1, -1)
        ).sum(dim=-1).to(dtype=dist_logits.dtype)
        return class_logits, box_deltas.flatten(start_dim=1), dist_logits


def _scale_roi_boxes(proposals, image_shapes, scale_factor: float):
    """Scale proposal boxes around their centers and clip to each image."""
    if scale_factor <= 0:
        raise ValueError("RoI scale factor must be positive")
    if scale_factor == 1.0:
        return proposals

    scaled = []
    for boxes, (height, width) in zip(proposals, image_shapes):
        if boxes.numel() == 0:
            scaled.append(boxes)
            continue
        centers = (boxes[:, :2] + boxes[:, 2:]) * 0.5
        half_sizes = (boxes[:, 2:] - boxes[:, :2]) * (0.5 * scale_factor)
        enlarged = torch.cat((centers - half_sizes, centers + half_sizes), dim=1)
        enlarged[:, 0::2].clamp_(min=0, max=width)
        enlarged[:, 1::2].clamp_(min=0, max=height)
        scaled.append(enlarged)
    return scaled


# =============================================================================
# Dynamic-K per GT
# =============================================================================
def _dynamic_k(wg, hg):
    sz = torch.sqrt((wg * hg).clamp(min=1.0))
    k_micro = torch.full_like(sz, RFLA_DYNAMIC_K_MICRO, dtype=torch.long)
    k_tiny  = torch.full_like(sz, RFLA_DYNAMIC_K_TINY,  dtype=torch.long)
    k_small = torch.full_like(sz, RFLA_DYNAMIC_K_SMALL, dtype=torch.long)
    k_large = torch.full_like(sz, RFLA_DYNAMIC_K_LARGE, dtype=torch.long)
    return torch.where(
        sz < 6.0, k_micro,
        torch.where(sz < 16.0, k_tiny,
        torch.where(sz < 64.0, k_small, k_large)))


# =============================================================================
# Hierarchical assignment (vectorized, giống v2 của notebook)
# =============================================================================
def _hierarchical_assignment(sim, xn, yn, wn, hn, xg, yg, wg, hg,
                             metric_fn=None, reliability_thr=16.0,
                             k=RFLA_K, beta=RFLA_BETA):
    N, M = sim.shape
    dev = sim.device
    matched_gt = torch.full((N,), -1, dtype=torch.long, device=dev)
    assigned_mask = torch.zeros(N, dtype=torch.bool, device=dev)
    if M == 0 or N == 0:
        return matched_gt

    dynamic_k = _dynamic_k(wg, hg).to(dev)
    base_k = min(k, N)
    max_k = int(dynamic_k.max().item())
    top_scores, top_idx = sim.topk(max_k, dim=0)

    keep_mask = torch.zeros(max_k, M, dtype=torch.bool, device=dev)
    keep_mask[:base_k] = True
    top1 = top_scores[0]
    quality_thr = (top1 * RFLA_QUALITY_RATIO).clamp(min=RFLA_MIN_SIM)
    dyn_k_mask = (torch.arange(max_k, device=dev).unsqueeze(1)
                  < dynamic_k.unsqueeze(0))
    keep_mask = keep_mask & dyn_k_mask
    extra_keep = (top_scores >= quality_thr.unsqueeze(0)) & dyn_k_mask
    keep_mask = keep_mask | extra_keep

    _, best_gt = sim.max(dim=1)
    anchor_ids = top_idx[keep_mask]
    gt_ids = keep_mask.nonzero(as_tuple=False)[:, 1]
    is_best = (gt_ids == best_gt[anchor_ids])
    anchor_ids = anchor_ids[is_best]
    gt_ids = gt_ids[is_best]
    matched_gt[anchor_ids] = gt_ids
    assigned_mask[anchor_ids] = True

    # Pass 2: mở rộng anchor hiệu dụng
    if metric_fn is not None:
        sim2 = metric_fn(xn, yn, wn * beta, hn * beta, xg, yg, wg, hg,
                         reliability_thr=reliability_thr)
        counts = torch.zeros(M, dtype=torch.long, device=dev)
        if assigned_mask.any():
            counts.scatter_add_(0,
                matched_gt.clamp(min=0)[assigned_mask],
                torch.ones(assigned_mask.sum(), dtype=torch.long, device=dev))
        has_room = counts < dynamic_k
        best2_score, best2_gt = sim2.max(dim=1)
        quality_thr2 = (sim2.max(dim=0).values * RFLA_QUALITY_RATIO).clamp(
            min=RFLA_MIN_SIM)
        ok = (best2_score >= quality_thr2[best2_gt]) & ~assigned_mask & has_room[best2_gt]
        if ok.any():
            matched_gt[ok] = best2_gt[ok]
            assigned_mask[ok] = True

    return matched_gt


# =============================================================================
# Metric RPN — label assignment via metric similarity (hierarchical top-k)
# =============================================================================
class MetricRPN(RegionProposalNetwork):
    def __init__(self, *args, metric_fn=None, reliability_thr=16.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_fn = metric_fn
        self.reliability_thr = reliability_thr

    def assign_targets_to_anchors(self, anchors, targets):
        labels_list, matched_boxes_list = [], []
        for anchors_img, targets_img in zip(anchors, targets):
            gt_boxes = targets_img["boxes"]
            dev = anchors_img.device
            if gt_boxes.numel() == 0:
                labels_list.append(
                    torch.zeros(len(anchors_img), dtype=torch.float32, device=dev))
                matched_boxes_list.append(torch.zeros_like(anchors_img))
                continue

            xn = (anchors_img[:, 0] + anchors_img[:, 2]) / 2.0
            yn = (anchors_img[:, 1] + anchors_img[:, 3]) / 2.0
            wn = (anchors_img[:, 2] - anchors_img[:, 0]).clamp(min=1.0)
            hn = (anchors_img[:, 3] - anchors_img[:, 1]).clamp(min=1.0)
            xg = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
            yg = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
            wg = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1.0)
            hg = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1.0)

            sim = self.metric_fn(xn, yn, wn, hn, xg, yg, wg, hg,
                                 reliability_thr=self.reliability_thr)
            mgt = _hierarchical_assignment(
                sim, xn, yn, wn, hn, xg, yg, wg, hg,
                metric_fn=self.metric_fn,
                reliability_thr=self.reliability_thr)

            lbl = torch.zeros(len(anchors_img), dtype=torch.float32, device=dev)
            lbl[mgt >= 0] = 1.0
            labels_list.append(lbl)
            matched_boxes_list.append(gt_boxes[mgt.clamp(min=0)])
        return labels_list, matched_boxes_list


# =============================================================================
# SAALW RPN — threshold-based label assignment via SAALWAssigner
# =============================================================================
class SAALWRPN(RegionProposalNetwork):
    """RPN with SAALWAssigner instead of hierarchical top-k.

    Uses threshold-based assignment: anchors with SA-ALW sim > pos_sim_thr
    become positive. Falls back to top-k if a GT has too few matches.
    """

    def __init__(self, *args,
                 saalw_assigner=None,  # SAALWAssigner instance
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.saalw_assigner = saalw_assigner

    def assign_targets_to_anchors(self, anchors, targets):
        labels_list, matched_boxes_list = [], []
        for anchors_img, targets_img in zip(anchors, targets):
            gt_boxes = targets_img["boxes"]
            dev = anchors_img.device
            if gt_boxes.numel() == 0:
                labels_list.append(
                    torch.zeros(len(anchors_img), dtype=torch.float32, device=dev))
                matched_boxes_list.append(torch.zeros_like(anchors_img))
                continue
            labels, matched = self.saalw_assigner(anchors_img, gt_boxes)
            lbl = labels.clone()
            lbl[lbl == -1] = 0.0  # candidate negatives -> 0 (RPN samples from these)
            labels_list.append(lbl)
            matched_boxes_list.append(matched)
        return labels_list, matched_boxes_list


# =============================================================================
# Metric-based box regression loss (with multi-type dispatch)
# =============================================================================
def _paired_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU for aligned box tensors [N,4] and [N,4]."""
    lt = torch.maximum(boxes1[:, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) *
             (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) *
             (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))
    return inter / (area1 + area2 - inter).clamp(min=EPS)


def _box_geometry_terms(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor):
    pred_w = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1.0)
    pred_h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1.0)
    gt_w = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1.0)
    gt_h = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1.0)
    xn = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
    yn = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0
    xg = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
    yg = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
    return xn, yn, pred_w, pred_h, xg, yg, gt_w, gt_h


def _metric_aux_loss(pred_boxes: torch.Tensor, gt_boxes: torch.Tensor,
                     metric_fn, reliability_thr: float) -> torch.Tensor:
    if metric_fn is None:
        return torch.zeros(1, device=pred_boxes.device).sum()
    xn, yn, wn, hn, xg, yg, wg, hg = _box_geometry_terms(pred_boxes, gt_boxes)
    sim = metric_fn(xn, yn, wn, hn, xg, yg, wg, hg,
                    reliability_thr=reliability_thr)
    return (1.0 - sim).mean()


def _side_aware_smooth_l1_loss(box_delta: torch.Tensor,
                               target_delta: torch.Tensor,
                               gt_boxes: torch.Tensor) -> torch.Tensor:
    """SABL-inspired delta loss with extra weight on tiny-object sides."""
    gt_w = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=2.0)
    gt_h = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=2.0)
    gt_size = (gt_w * gt_h).sqrt()
    tiny_weight = (16.0 / gt_size).clamp(min=1.0, max=4.0)
    raw = F.smooth_l1_loss(
        box_delta, target_delta, beta=1.0, reduction="none")
    return (raw * tiny_weight[:, None]).mean()


def _cbl_localization_loss(
    distribution_logits: torch.Tensor,
    target_deltas: torch.Tensor,
    grid: torch.Tensor,
    uncertainty_weight: float,
) -> torch.Tensor:
    """Two-hot confidence loss plus entropy-matching uncertainty loss."""
    if distribution_logits.ndim != 3:
        raise ValueError("CBL logits must have shape [positive_rois, 4, bins]")
    if distribution_logits.shape[-1] != grid.numel():
        raise ValueError("CBL logits and grid have incompatible bin counts")

    logits = distribution_logits.float()
    grid = grid.to(device=logits.device, dtype=logits.dtype)
    targets = target_deltas.float().clamp(
        min=float(grid[0]), max=float(grid[-1]))

    right_idx = torch.searchsorted(
        grid, targets.contiguous(), right=True).clamp(1, grid.numel() - 1)
    left_idx = right_idx - 1
    left_grid = grid[left_idx]
    right_grid = grid[right_idx]
    right_weight = (
        (targets - left_grid) / (right_grid - left_grid).clamp(min=EPS)
    ).clamp(0.0, 1.0)
    left_weight = 1.0 - right_weight

    log_probs = F.log_softmax(logits, dim=-1)
    left_log_prob = log_probs.gather(-1, left_idx.unsqueeze(-1)).squeeze(-1)
    right_log_prob = log_probs.gather(-1, right_idx.unsqueeze(-1)).squeeze(-1)
    confidence_loss = -(
        left_weight * left_log_prob + right_weight * right_log_prob
    ).mean()

    if uncertainty_weight <= 0:
        return confidence_loss

    probs = log_probs.exp()
    prediction_entropy = -(probs * log_probs).sum(dim=-1)
    target_entropy = -(
        left_weight * left_weight.clamp(min=EPS).log()
        + right_weight * right_weight.clamp(min=EPS).log()
    )
    uncertainty_loss = (
        prediction_entropy - target_entropy
    ).abs().mean()
    return confidence_loss + uncertainty_weight * uncertainty_loss


def _quality_focal_loss(
    class_logits: torch.Tensor,
    labels: torch.Tensor,
    quality_targets: torch.Tensor,
    beta: float = 2.0,
) -> torch.Tensor:
    """Quality Focal Loss over foreground classes with IoU soft targets."""
    if beta < 0:
        raise ValueError("Quality Focal Loss beta must be non-negative")
    if class_logits.ndim != 2 or class_logits.shape[1] < 2:
        raise ValueError("QFL expects logits for background plus foreground classes")
    if labels.shape != quality_targets.shape:
        raise ValueError("QFL labels and quality targets must have the same shape")
    if labels.numel() != class_logits.shape[0]:
        raise ValueError("QFL targets and logits have incompatible batch sizes")

    foreground_logits = class_logits[:, 1:].float()
    probabilities = foreground_logits.sigmoid()
    zero_targets = torch.zeros_like(foreground_logits)
    loss = F.binary_cross_entropy_with_logits(
        foreground_logits, zero_targets, reduction="none"
    ) * probabilities.pow(beta)

    positive = torch.where(labels > 0)[0]
    if positive.numel() > 0:
        foreground_labels = labels[positive] - 1
        if foreground_labels.max() >= foreground_logits.shape[1]:
            raise ValueError("QFL foreground label exceeds classifier dimensions")
        targets = quality_targets[positive].float().clamp(0.0, 1.0)
        positive_logits = foreground_logits[positive, foreground_labels]
        positive_probabilities = probabilities[positive, foreground_labels]
        loss[positive, foreground_labels] = F.binary_cross_entropy_with_logits(
            positive_logits, targets, reduction="none"
        ) * (targets - positive_probabilities).abs().pow(beta)

    return loss.sum(dim=1).mean()


class _RankSortFunction(torch.autograd.Function):
    """Device-agnostic Rank & Sort identity-update autograd function."""

    @staticmethod
    def forward(ctx, logits, targets, delta=0.5, eps=1e-10):
        logits = logits.reshape(-1)
        targets = targets.reshape(-1)
        classification_grads = torch.zeros_like(logits)

        foreground_mask = targets > 0
        foreground_logits = logits[foreground_mask]
        foreground_targets = targets[foreground_mask]
        foreground_count = foreground_logits.numel()
        if foreground_count == 0:
            ctx.save_for_backward(classification_grads)
            zero = logits.new_zeros(())
            return zero, zero

        threshold = foreground_logits.min() - delta
        relevant_background_mask = (targets == 0) & (logits >= threshold)
        relevant_background_logits = logits[relevant_background_mask]
        relevant_background_grad = torch.zeros_like(relevant_background_logits)
        ranking_error = torch.zeros_like(foreground_logits)
        sorting_error = torch.zeros_like(foreground_logits)
        foreground_grad = torch.zeros_like(foreground_logits)

        order = torch.argsort(foreground_logits)
        for index in order:
            foreground_relations = foreground_logits - foreground_logits[index]
            background_relations = (
                relevant_background_logits - foreground_logits[index]
            )
            if delta > 0:
                foreground_relations = torch.clamp(
                    foreground_relations / (2 * delta) + 0.5, min=0, max=1
                )
                background_relations = torch.clamp(
                    background_relations / (2 * delta) + 0.5, min=0, max=1
                )
            else:
                foreground_relations = (foreground_relations >= 0).to(logits.dtype)
                background_relations = (background_relations >= 0).to(logits.dtype)

            positive_rank = foreground_relations.sum()
            false_positive_count = background_relations.sum()
            rank = positive_rank + false_positive_count
            ranking_error[index] = false_positive_count / rank.clamp(min=eps)

            current_sorting_error = (
                foreground_relations * (1 - foreground_targets)
            ).sum() / positive_rank.clamp(min=eps)
            iou_relations = foreground_targets >= foreground_targets[index]
            target_sorted_order = iou_relations.to(logits.dtype) * foreground_relations
            target_positive_rank = target_sorted_order.sum()
            target_sorting_error = (
                target_sorted_order * (1 - foreground_targets)
            ).sum() / target_positive_rank.clamp(min=eps)
            sorting_error[index] = current_sorting_error - target_sorting_error

            if false_positive_count > eps:
                foreground_grad[index] -= ranking_error[index]
                relevant_background_grad += (
                    background_relations
                    * (ranking_error[index] / false_positive_count)
                )

            missorted = (~iou_relations).to(logits.dtype) * foreground_relations
            sorting_denom = missorted.sum()
            if sorting_denom > eps:
                foreground_grad[index] -= sorting_error[index]
                foreground_grad += missorted * (
                    sorting_error[index] / sorting_denom
                )

        classification_grads[foreground_mask] = (
            foreground_grad / foreground_count
        )
        classification_grads[relevant_background_mask] = (
            relevant_background_grad / foreground_count
        )
        ctx.save_for_backward(classification_grads)
        return ranking_error.mean(), sorting_error.mean()

    @staticmethod
    def backward(ctx, ranking_grad, sorting_grad):
        del sorting_grad
        (classification_grads,) = ctx.saved_tensors
        # The saved identity update already contains ranking and sorting terms.
        return classification_grads * ranking_grad, None, None, None


def _rank_sort_loss(
    class_logits: torch.Tensor,
    labels: torch.Tensor,
    quality_targets: torch.Tensor,
    delta: float = 0.5,
) -> torch.Tensor:
    """Rank foreground classes and sort positives by paired localization IoU."""
    if delta < 0:
        raise ValueError("Rank & Sort delta must be non-negative")
    if class_logits.ndim != 2 or class_logits.shape[1] < 2:
        raise ValueError(
            "Rank & Sort expects logits for background plus foreground classes"
        )
    if labels.shape != quality_targets.shape:
        raise ValueError(
            "Rank & Sort labels and quality targets must have the same shape"
        )
    if labels.numel() != class_logits.shape[0]:
        raise ValueError(
            "Rank & Sort targets and logits have incompatible batch sizes"
        )

    foreground_logits = class_logits[:, 1:].float()
    targets = torch.zeros_like(foreground_logits)
    positive = torch.where(labels > 0)[0]
    if positive.numel() > 0:
        foreground_labels = labels[positive] - 1
        if foreground_labels.max() >= foreground_logits.shape[1]:
            raise ValueError(
                "Rank & Sort foreground label exceeds classifier dimensions"
            )
        targets[positive, foreground_labels] = quality_targets[positive].float().clamp(
            0.0, 1.0
        )

    ranking_loss, sorting_loss = _RankSortFunction.apply(
        foreground_logits.reshape(-1), targets.reshape(-1), delta
    )
    return ranking_loss + sorting_loss


def _metric_box_loss(class_logits, box_regression, labels, regression_targets,
                     box_coder, metric_fn, reliability_thr, metric_loss_weight,
                     proposals, current_epoch=1, box_loss_type=None,
                     box_loss_warmup_epochs=BOX_LOSS_WARMUP_EPOCHS,
                     quality_logits=None, quality_loss_weight=0.0,
                     use_quality_focal=False, quality_focal_beta=2.0,
                     use_rank_sort=False, rank_sort_delta=0.5,
                     distribution_logits=None, cbl_grid=None,
                     cbl_um_weight=CBL_UM_WEIGHT):
    """Replacement for torchvision's fastrcnn_loss with multi-loss dispatch.

    Args:
        current_epoch: used for warmup (pure metric loss first N epochs, then ramp)
        box_loss_type: "metric", "smooth_l1", "side_smooth_l1", "ciou",
            "diou", or "cbl"
    """
    if box_loss_type is None:
        box_loss_type = BOX_LOSS_TYPE
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = (
        class_logits.sum() * 0.0
        if use_quality_focal or use_rank_sort
        else F.cross_entropy(class_logits, labels)
    )

    sampled_pos_inds = torch.where(labels > 0)[0]
    if sampled_pos_inds.numel() == 0:
        if use_quality_focal:
            quality_targets = class_logits.new_zeros(labels.shape)
            classification_loss = _quality_focal_loss(
                class_logits, labels, quality_targets, quality_focal_beta)
        elif use_rank_sort:
            quality_targets = class_logits.new_zeros(labels.shape)
            classification_loss = _rank_sort_loss(
                class_logits, labels, quality_targets, rank_sort_delta)
        zero = torch.zeros(1, device=class_logits.device).sum()
        return classification_loss, zero, zero

    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, num_classes, 4)
    K = len(sampled_pos_inds)
    box_regression_pos = box_regression[sampled_pos_inds]
    labels_pos = labels[sampled_pos_inds]
    targets_deltas = regression_targets[sampled_pos_inds]

    proposals_flat = torch.cat(proposals, dim=0)
    proposals_pos = proposals_flat[sampled_pos_inds]
    decoded_gt = box_coder.decode(targets_deltas, [proposals_pos])
    gt_boxes_for_quality = decoded_gt[:, 0, :]
    pred_boxes_for_quality = None

    # ── Determine loss type ──
    loss_type = box_loss_type

    # ── Warmup: pure metric loss for first warmup_epochs ──
    if (loss_type != "metric" and box_loss_warmup_epochs > 0 and
            current_epoch <= box_loss_warmup_epochs):
        loss_type = "metric"

    # ── Compute primary box loss ──
    if loss_type == "metric":
        # Current Gaussian similarity loss
        box_reg_flat = box_regression_pos.reshape(K, num_classes * 4)
        decoded = box_coder.decode(box_reg_flat, [proposals_pos])
        pred_boxes = decoded[torch.arange(K, device=decoded.device), labels_pos]
        gt_boxes = gt_boxes_for_quality
        pred_boxes_for_quality = pred_boxes

        box_loss = _metric_aux_loss(
            pred_boxes, gt_boxes, metric_fn, reliability_thr) * metric_loss_weight

    elif loss_type in ("smooth_l1", "side_smooth_l1"):
        # Standard Smooth-L1 on delta space (mirrors RFLA's AP75=18.8)
        # Select the regression output for the positive class
        box_reg_pos_per_class = box_regression_pos[
            torch.arange(K, device=box_regression_pos.device), labels_pos]  # (K, 4)
        delta_loss = F.smooth_l1_loss(
            box_reg_pos_per_class, targets_deltas, beta=1.0)
        box_reg_flat = box_regression_pos.reshape(K, num_classes * 4)
        decoded = box_coder.decode(box_reg_flat, [proposals_pos])
        pred_boxes = decoded[torch.arange(K, device=decoded.device), labels_pos]
        gt_boxes = gt_boxes_for_quality
        pred_boxes_for_quality = pred_boxes
        metric_aux = _metric_aux_loss(
            pred_boxes, gt_boxes, metric_fn, reliability_thr)
        if loss_type == "side_smooth_l1":
            side_loss = _side_aware_smooth_l1_loss(
                box_reg_pos_per_class, targets_deltas, gt_boxes)
            box_loss = side_loss + BOX_LOSS_METRIC_WEIGHT * metric_aux
        else:
            # Add auxiliary metric loss (small weight) to preserve micro stability
            box_loss = delta_loss + BOX_LOSS_METRIC_WEIGHT * metric_aux

    elif loss_type == "cbl":
        if distribution_logits is None or cbl_grid is None:
            raise ValueError("CBL loss requires distribution logits and a grid")
        dist_pos = distribution_logits[
            sampled_pos_inds, labels_pos]  # [K, 4, bins]
        box_loss = _cbl_localization_loss(
            dist_pos, targets_deltas, cbl_grid, cbl_um_weight)
        box_reg_flat = box_regression_pos.reshape(K, num_classes * 4)
        decoded = box_coder.decode(box_reg_flat, [proposals_pos])
        pred_boxes_for_quality = decoded[
            torch.arange(K, device=decoded.device), labels_pos]

    elif loss_type in ("ciou", "diou"):
        # Entire CIoU/DIoU block must run in float32 outside autocast.
        # Under AMP float16, torchvision's CIoU kernel can segfault on
        # tiny boxes (2-8 px) due to degenerate float16 area computations.
        with torch.amp.autocast("cuda", enabled=False):
            # Decode boxes for IoU-based loss (force float32)
            box_reg_flat = box_regression_pos.float().reshape(K, num_classes * 4)
            decoded = box_coder.decode(box_reg_flat, [proposals_pos.float()])  
            pred_boxes = decoded[torch.arange(K, device=decoded.device), labels_pos]
            decoded_gt = box_coder.decode(targets_deltas.float(), [proposals_pos.float()])
            gt_boxes = decoded_gt[:, 0, :]
            pred_boxes_for_quality = pred_boxes.to(dtype=box_regression.dtype)
            gt_boxes_for_quality = gt_boxes.to(dtype=box_regression.dtype)

            # Clamp degenerate boxes
            pred_w = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=2.0)
            pred_h = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=2.0)
            pred_boxes = torch.stack([
                pred_boxes[:, 0], pred_boxes[:, 1],
                pred_boxes[:, 0] + pred_w, pred_boxes[:, 1] + pred_h,
            ], dim=1)
            gt_w = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=2.0)
            gt_h = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=2.0)
            gt_boxes = torch.stack([
                gt_boxes[:, 0], gt_boxes[:, 1],
                gt_boxes[:, 0] + gt_w, gt_boxes[:, 1] + gt_h,
            ], dim=1)

            # Filter: remove boxes with near-zero area
            pred_area = pred_w * pred_h
            gt_area = gt_w * gt_h
            valid = (pred_area >= 4.0) & (gt_area >= 4.0)
            if valid.sum() < 1:
                iou_loss = torch.tensor(0.0, device=pred_boxes.device)
                metric_aux_val = torch.tensor(0.0, device=pred_boxes.device)
            else:
                pred_boxes_f = pred_boxes[valid]
                gt_boxes_f = gt_boxes[valid]

                if loss_type == "ciou":
                    iou_loss = complete_box_iou_loss(
                        pred_boxes_f, gt_boxes_f, reduction="mean")
                else:
                    iou_loss = distance_box_iou_loss(
                        pred_boxes_f, gt_boxes_f, reduction="mean")

                if not torch.isfinite(iou_loss):
                    iou_loss = torch.tensor(0.0, device=iou_loss.device)
                iou_loss = iou_loss.clamp(max=5.0)

                # Auxiliary metric loss
                xn = (pred_boxes_f[:, 0] + pred_boxes_f[:, 2]) / 2.0
                yn = (pred_boxes_f[:, 1] + pred_boxes_f[:, 3]) / 2.0
                wn = (pred_boxes_f[:, 2] - pred_boxes_f[:, 0]).clamp(min=1.0)
                hn = (pred_boxes_f[:, 3] - pred_boxes_f[:, 1]).clamp(min=1.0)
                xg = (gt_boxes_f[:, 0] + gt_boxes_f[:, 2]) / 2.0
                yg = (gt_boxes_f[:, 1] + gt_boxes_f[:, 3]) / 2.0
                wg = (gt_boxes_f[:, 2] - gt_boxes_f[:, 0]).clamp(min=1.0)
                hg = (gt_boxes_f[:, 3] - gt_boxes_f[:, 1]).clamp(min=1.0)
                sim = metric_fn(xn, yn, wn, hn, xg, yg, wg, hg,
                                 reliability_thr=reliability_thr)
                metric_aux_val = (1.0 - sim).mean()

        # Cast back to autocast dtype for box_loss
        iou_loss = iou_loss.to(dtype=pred_boxes.dtype)
        metric_aux = metric_aux_val.to(dtype=pred_boxes.dtype)
        box_loss = iou_loss + BOX_LOSS_METRIC_WEIGHT * metric_aux

    else:
        raise ValueError(f"Unknown BOX_LOSS_TYPE: {loss_type}")

    quality_loss = torch.zeros(1, device=class_logits.device).sum()
    paired_quality_targets = None
    if use_quality_focal or use_rank_sort or (
            quality_logits is not None and quality_loss_weight > 0):
        if pred_boxes_for_quality is None:
            box_reg_flat = box_regression_pos.reshape(K, num_classes * 4)
            decoded = box_coder.decode(box_reg_flat, [proposals_pos])
            pred_boxes_for_quality = decoded[
                torch.arange(K, device=decoded.device), labels_pos]
        paired_quality_targets = _paired_iou(
            pred_boxes_for_quality.detach().float(),
            gt_boxes_for_quality.detach().float(),
        ).clamp(0, 1)

    if use_quality_focal:
        quality_targets = class_logits.new_zeros(labels.shape)
        quality_targets[sampled_pos_inds] = paired_quality_targets.to(
            dtype=class_logits.dtype)
        classification_loss = _quality_focal_loss(
            class_logits, labels, quality_targets, quality_focal_beta)

    if use_rank_sort:
        quality_targets = class_logits.new_zeros(labels.shape)
        quality_targets[sampled_pos_inds] = paired_quality_targets.to(
            dtype=class_logits.dtype)
        classification_loss = _rank_sort_loss(
            class_logits, labels, quality_targets, rank_sort_delta)

    if quality_logits is not None and quality_loss_weight > 0:
        q_targets = paired_quality_targets.to(dtype=quality_logits.dtype)
        q_pred = quality_logits[sampled_pos_inds, labels_pos]
        quality_loss = F.binary_cross_entropy_with_logits(q_pred, q_targets)
        quality_loss = quality_loss * quality_loss_weight

    return classification_loss, box_loss, quality_loss


def _postprocess_detections_with_quality(
    roi_heads, class_logits, box_regression, quality_logits, proposals, image_shapes
):
    """Postprocess detections with score = class_prob * predicted localization quality."""
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = roi_heads.box_coder.decode(box_regression, proposals)
    pred_scores = F.softmax(class_logits, -1) * torch.sigmoid(quality_logits)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes, all_scores, all_labels = [], [], []
    for boxes, scores, image_shape in zip(pred_boxes_list, pred_scores_list, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        boxes = boxes[:, 1:]
        scores = scores[:, 1:]
        labels = labels[:, 1:]

        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        inds = torch.where(scores > roi_heads.score_thresh)[0]
        boxes, scores, labels = boxes[inds], scores[inds], labels[inds]

        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]

        keep = box_ops.batched_nms(boxes, scores, labels, roi_heads.nms_thresh)
        keep = keep[: roi_heads.detections_per_img]
        all_boxes.append(boxes[keep])
        all_scores.append(scores[keep])
        all_labels.append(labels[keep])

    return all_boxes, all_scores, all_labels


def _postprocess_detections_with_joint_scores(
    roi_heads, class_logits, box_regression, proposals, image_shapes
):
    """Postprocess detections using QFL's joint class-localization score."""
    device = class_logits.device
    num_classes = class_logits.shape[-1]

    boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]
    pred_boxes = roi_heads.box_coder.decode(box_regression, proposals)
    pred_scores = torch.sigmoid(class_logits)

    pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
    pred_scores_list = pred_scores.split(boxes_per_image, 0)

    all_boxes, all_scores, all_labels = [], [], []
    for boxes, scores, image_shape in zip(
            pred_boxes_list, pred_scores_list, image_shapes):
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
        labels = torch.arange(num_classes, device=device)
        labels = labels.view(1, -1).expand_as(scores)

        boxes = boxes[:, 1:].reshape(-1, 4)
        scores = scores[:, 1:].reshape(-1)
        labels = labels[:, 1:].reshape(-1)

        keep = torch.where(scores > roi_heads.score_thresh)[0]
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        keep = box_ops.batched_nms(
            boxes, scores, labels, roi_heads.nms_thresh)
        keep = keep[:roi_heads.detections_per_img]
        all_boxes.append(boxes[keep])
        all_scores.append(scores[keep])
        all_labels.append(labels[keep])

    return all_boxes, all_scores, all_labels


def _iteratively_refine_cbl_detections(
    roi_heads,
    features,
    boxes,
    scores,
    labels,
    image_shapes,
    steps,
    blend,
    score_threshold,
):
    """Reapply the trained CBL regressor while preserving labels and scores."""
    refine_box_head = getattr(
        roi_heads, "refine_box_head", roi_heads.box_head)
    refine_box_predictor = getattr(
        roi_heads, "refine_box_predictor", roi_heads.box_predictor)
    current_boxes = boxes
    for _ in range(steps):
        if not any(boxes_per_image.numel() for boxes_per_image in current_boxes):
            break
        pooled = roi_heads.box_roi_pool(
            features, current_boxes, image_shapes)
        box_features = refine_box_head(pooled)
        predictor_out = refine_box_predictor(box_features)
        if (not getattr(refine_box_predictor, "is_distributional", False)
                or len(predictor_out) != 3):
            raise RuntimeError(
                "Iterative CBL refinement requires a distributional predictor")

        _, box_regression, _ = predictor_out
        decoded = roi_heads.box_coder.decode(box_regression, current_boxes)
        decoded_per_image = decoded.split(
            [len(boxes_per_image) for boxes_per_image in current_boxes], 0)

        refined_boxes = []
        for (
            decoded_boxes,
            boxes_per_image,
            scores_per_image,
            labels_per_image,
            image_shape,
        ) in zip(decoded_per_image, current_boxes, scores, labels, image_shapes):
            if decoded_boxes.numel() == 0:
                refined_boxes.append(decoded_boxes.reshape(0, 4))
                continue
            row_ids = torch.arange(
                len(labels_per_image), device=decoded_boxes.device)
            selected = decoded_boxes[row_ids, labels_per_image]
            selected = (
                boxes_per_image + blend * (selected - boxes_per_image)
            )
            if score_threshold > 0:
                selected = torch.where(
                    (scores_per_image >= score_threshold).unsqueeze(1),
                    selected,
                    boxes_per_image,
                )
            refined_boxes.append(
                box_ops.clip_boxes_to_image(selected, image_shape))
        current_boxes = refined_boxes

    final_boxes, final_scores, final_labels = [], [], []
    for boxes_per_image, scores_per_image, labels_per_image in zip(
            current_boxes, scores, labels):
        finite = torch.isfinite(boxes_per_image).all(dim=1)
        boxes_per_image = boxes_per_image[finite]
        scores_per_image = scores_per_image[finite]
        labels_per_image = labels_per_image[finite]
        keep = box_ops.remove_small_boxes(boxes_per_image, min_size=1e-2)
        boxes_per_image = boxes_per_image[keep]
        scores_per_image = scores_per_image[keep]
        labels_per_image = labels_per_image[keep]
        keep = box_ops.batched_nms(
            boxes_per_image,
            scores_per_image,
            labels_per_image,
            roi_heads.nms_thresh,
        )
        keep = keep[:roi_heads.detections_per_img]
        final_boxes.append(boxes_per_image[keep])
        final_scores.append(scores_per_image[keep])
        final_labels.append(labels_per_image[keep])
    return final_boxes, final_scores, final_labels


def _iterative_cbl_training_loss(
    roi_heads,
    features,
    proposals,
    labels,
    regression_targets,
    box_regression,
    image_shapes,
    uncertainty_weight,
    loss_weight,
):
    """Train a CBL head on detached first-pass box proposals."""
    if loss_weight <= 0:
        return box_regression.sum() * 0.0

    counts = [len(proposals_per_image) for proposals_per_image in proposals]
    num_classes = box_regression.shape[1] // 4
    box_regression_per_image = box_regression.reshape(
        -1, num_classes, 4).split(counts, 0)

    refined_proposals = []
    refined_gt_boxes = []
    refined_labels = []
    for (
        proposals_per_image,
        labels_per_image,
        targets_per_image,
        regression_per_image,
        image_shape,
    ) in zip(
        proposals,
        labels,
        regression_targets,
        box_regression_per_image,
        image_shapes,
    ):
        positive = torch.where(labels_per_image > 0)[0]
        if positive.numel() == 0:
            refined_proposals.append(proposals_per_image.new_zeros((0, 4)))
            refined_gt_boxes.append(proposals_per_image.new_zeros((0, 4)))
            refined_labels.append(labels_per_image.new_zeros((0,)))
            continue

        positive_proposals = proposals_per_image[positive]
        positive_labels = labels_per_image[positive]
        first_pass_deltas = regression_per_image[
            positive, positive_labels]
        first_pass_boxes = roi_heads.box_coder.decode(
            first_pass_deltas, [positive_proposals])[:, 0]
        gt_boxes = roi_heads.box_coder.decode(
            targets_per_image[positive], [positive_proposals])[:, 0]

        first_pass_boxes = box_ops.clip_boxes_to_image(
            first_pass_boxes.detach(), image_shape)
        gt_boxes = box_ops.clip_boxes_to_image(
            gt_boxes.detach(), image_shape)
        valid = (
            torch.isfinite(first_pass_boxes).all(dim=1)
            & torch.isfinite(gt_boxes).all(dim=1)
            & ((first_pass_boxes[:, 2:] - first_pass_boxes[:, :2]).min(dim=1).values
               > 1e-2)
        )
        refined_proposals.append(first_pass_boxes[valid])
        refined_gt_boxes.append(gt_boxes[valid])
        refined_labels.append(positive_labels[valid])

    if not any(len(boxes) for boxes in refined_proposals):
        return box_regression.sum() * 0.0

    refine_box_head = getattr(
        roi_heads, "refine_box_head", roi_heads.box_head)
    refine_box_predictor = getattr(
        roi_heads, "refine_box_predictor", roi_heads.box_predictor)
    refined_features = roi_heads.box_roi_pool(
        features, refined_proposals, image_shapes)
    refined_features = refine_box_head(refined_features)
    predictor_out = refine_box_predictor(refined_features)
    if (not getattr(refine_box_predictor, "is_distributional", False)
            or len(predictor_out) != 3):
        raise RuntimeError(
            "Iterative CBL training requires a distributional predictor")

    _, _, refined_distribution_logits = predictor_out
    labels_flat = torch.cat(refined_labels, dim=0)
    rows = torch.arange(
        len(labels_flat), device=refined_distribution_logits.device)
    selected_logits = refined_distribution_logits[rows, labels_flat]
    refined_targets = torch.cat(
        roi_heads.box_coder.encode(refined_gt_boxes, refined_proposals),
        dim=0,
    )
    return loss_weight * _cbl_localization_loss(
        selected_logits,
        refined_targets,
        refine_box_predictor.cbl_grid,
        uncertainty_weight,
    )


def _wrap_roi_forward_for_metric_loss(roi_heads, metric_fn, reliability_thr,
                                       metric_loss_weight=1.0,
                                       box_loss_type="metric",
                                       box_loss_warmup_epochs=BOX_LOSS_WARMUP_EPOCHS,
                                       quality_loss_weight=0.0,
                                       use_quality_focal=False,
                                       quality_focal_beta=2.0,
                                       use_rank_sort=False,
                                       rank_sort_delta=0.5,
                                       use_double_head=False,
                                       double_head_reg_roi_scale=1.3,
                                       cbl_refine_steps=0,
                                       cbl_refine_blend=1.0,
                                       cbl_refine_score_threshold=0.0,
                                       cbl_refine_train_weight=0.0,
                                       cbl_refine_separate_head=False,
                                       cbl_um_weight=CBL_UM_WEIGHT):
    """Monkey-patch RoIHeads.forward to replace fastrcnn_loss with metric loss."""
    original_forward = roi_heads.forward
    # Store on roi_heads for access during training
    roi_heads._box_loss_type = box_loss_type
    roi_heads._box_loss_warmup_epochs = box_loss_warmup_epochs
    roi_heads._quality_loss_weight = quality_loss_weight
    roi_heads._use_quality_focal = use_quality_focal
    roi_heads._quality_focal_beta = quality_focal_beta
    roi_heads._use_rank_sort = use_rank_sort
    roi_heads._rank_sort_delta = rank_sort_delta
    roi_heads._use_double_head = use_double_head
    roi_heads._double_head_reg_roi_scale = double_head_reg_roi_scale
    roi_heads._cbl_refine_steps = cbl_refine_steps
    roi_heads._cbl_refine_blend = cbl_refine_blend
    roi_heads._cbl_refine_score_threshold = cbl_refine_score_threshold
    roi_heads._cbl_refine_train_weight = cbl_refine_train_weight
    roi_heads._cbl_refine_separate_head = cbl_refine_separate_head
    roi_heads._cbl_um_weight = cbl_um_weight

    def patched_forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None and self.training:
            proposals_sampled, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)
        else:
            labels = None; regression_targets = None; matched_idxs = None
            proposals_sampled = proposals

        box_features = self.box_roi_pool(
            features, proposals_sampled, image_shapes)
        cls_features = self.box_head(box_features)
        if getattr(self, "_use_double_head", False):
            regression_proposals = _scale_roi_boxes(
                proposals_sampled,
                image_shapes,
                getattr(self, "_double_head_reg_roi_scale", 1.3),
            )
            reg_features = self.box_roi_pool(
                features, regression_proposals, image_shapes)
            predictor_out = self.box_predictor(cls_features, reg_features)
        else:
            predictor_out = self.box_predictor(cls_features)
        distribution_logits = None
        if getattr(self.box_predictor, "is_distributional", False):
            class_logits, box_regression, distribution_logits = predictor_out
            quality_logits = None
        elif len(predictor_out) == 3:
            class_logits, box_regression, quality_logits = predictor_out
        else:
            class_logits, box_regression = predictor_out
            quality_logits = None

        result = []
        losses = {}
        if self.training:
            current_epoch = getattr(self, '_current_epoch', 1)
            box_loss_type = getattr(self, '_box_loss_type', BOX_LOSS_TYPE)
            box_loss_warmup_epochs = getattr(
                self, '_box_loss_warmup_epochs', BOX_LOSS_WARMUP_EPOCHS)
            quality_loss_weight = getattr(self, '_quality_loss_weight', 0.0)
            use_quality_focal = getattr(self, '_use_quality_focal', False)
            quality_focal_beta = getattr(
                self, '_quality_focal_beta', 2.0)
            use_rank_sort = getattr(self, '_use_rank_sort', False)
            rank_sort_delta = getattr(self, '_rank_sort_delta', 0.5)
            cbl_um_weight = getattr(self, '_cbl_um_weight', CBL_UM_WEIGHT)
            loss_classifier, loss_box_reg, loss_quality = _metric_box_loss(
                class_logits, box_regression, labels, regression_targets,
                self.box_coder, metric_fn, reliability_thr, metric_loss_weight,
                proposals_sampled, current_epoch=current_epoch,
                box_loss_type=box_loss_type,
                box_loss_warmup_epochs=box_loss_warmup_epochs,
                quality_logits=quality_logits,
                quality_loss_weight=quality_loss_weight,
                use_quality_focal=use_quality_focal,
                quality_focal_beta=quality_focal_beta,
                use_rank_sort=use_rank_sort,
                rank_sort_delta=rank_sort_delta,
                distribution_logits=distribution_logits,
                cbl_grid=getattr(self.box_predictor, "cbl_grid", None),
                cbl_um_weight=cbl_um_weight)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
            refine_train_weight = getattr(
                self, "_cbl_refine_train_weight", 0.0)
            if refine_train_weight > 0:
                losses["loss_box_refine"] = _iterative_cbl_training_loss(
                    self,
                    features,
                    proposals_sampled,
                    labels,
                    regression_targets,
                    box_regression,
                    image_shapes,
                    cbl_um_weight,
                    refine_train_weight,
                )
            if quality_logits is not None and quality_loss_weight > 0:
                losses["loss_quality"] = loss_quality
        else:
            if (getattr(self, '_use_quality_focal', False) or
                    getattr(self, '_use_rank_sort', False)):
                boxes, scores, labels_out = \
                    _postprocess_detections_with_joint_scores(
                        self, class_logits, box_regression,
                        proposals, image_shapes)
            elif quality_logits is not None:
                boxes, scores, labels_out = _postprocess_detections_with_quality(
                    self, class_logits, box_regression, quality_logits,
                    proposals, image_shapes)
            else:
                boxes, scores, labels_out = self.postprocess_detections(
                    class_logits, box_regression, proposals, image_shapes)
            refine_steps = getattr(self, "_cbl_refine_steps", 0)
            if refine_steps > 0:
                boxes, scores, labels_out = _iteratively_refine_cbl_detections(
                    self,
                    features,
                    boxes,
                    scores,
                    labels_out,
                    image_shapes,
                    refine_steps,
                    getattr(self, "_cbl_refine_blend", 1.0),
                    getattr(self, "_cbl_refine_score_threshold", 0.0),
                )
            for i in range(len(boxes)):
                result.append({"boxes": boxes[i], "labels": labels_out[i], "scores": scores[i]})

        if self.has_mask():
            mask_proposals = [p["boxes"] for p in result]
            if self.training:
                if matched_idxs is not None:
                    pos_matched_idxs = matched_idxs[matched_idxs >= 0]
                    mask_proposals = mask_proposals[:len(pos_matched_idxs)]
                mask_losses = self.mask_roi_pool(features, mask_proposals, image_shapes)
                losses.update(mask_losses)
            else:
                mask_logits = self.mask_roi_pool(features, mask_proposals, image_shapes)
                for i, logits in enumerate(mask_logits):
                    result[i]["masks"] = logits

        if self.has_keypoint():
            keypoint_proposals = [p["boxes"] for p in result]
            if self.training:
                keypoint_losses = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
                losses.update(keypoint_losses)
            else:
                keypoint_logits = self.keypoint_roi_pool(features, keypoint_proposals, image_shapes)
                for i, logits in enumerate(keypoint_logits):
                    result[i]["keypoints"] = logits

        return result, losses

    roi_heads.forward = patched_forward.__get__(roi_heads, type(roi_heads))


# =============================================================================
# Metric-based NMS (hard suppression)
# =============================================================================
def metric_nms(boxes, scores, metric_fn=None, reliability_thr=16.0,
               iou_thresh=NMS_METRIC_THRESH):
    """NMS where 'overlap' is metric_similarity > thresh (hard suppression)."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    if metric_fn is None:
        return batched_nms(boxes, scores,
                           torch.zeros(len(boxes), dtype=torch.long, device=boxes.device),
                           iou_thresh)
    xn = (boxes[:, 0] + boxes[:, 2]) / 2.0
    yn = (boxes[:, 1] + boxes[:, 3]) / 2.0
    wn = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
    hn = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
    sim = metric_fn(xn, yn, wn, hn, xn, yn, wn, hn,
                    reliability_thr=reliability_thr)
    order = scores.argsort(descending=True)
    keep = []
    suppressed = torch.zeros(len(boxes), dtype=torch.bool, device=boxes.device)
    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx.item())
        rest = order[(order != idx) & (~suppressed[order])]
        if len(rest) > 0:
            sup = sim[idx, rest] > iou_thresh
            suppressed[rest[sup]] = True
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


# =============================================================================
# ALW-Soft-NMS: decay scores instead of hard suppression
# =============================================================================
def soft_metric_nms(boxes, scores, metric_fn=None, reliability_thr=16.0,
                    sim_thresh: float = 0.3, score_thresh: float = 0.001,
                    decay: str = "linear"):
    """Soft-NMS using metric similarity instead of IoU.

    Algorithm (Bodla et al. 2017, adapted):
      1. Sort boxes by score descending
      2. For each box i (highest remaining):
         - Keep it at its current score
         - For each remaining box j with sim(i,j) > sim_thresh:
           score_j *= (1 - sim(i,j))           [linear decay]
           or score_j *= exp(-sim(i,j)^2/sigma) [gaussian decay]
      3. Filter kept boxes with score >= score_thresh

    Args:
        boxes: [N, 4]
        scores: [N]
        metric_fn: ALW metric function (returns exp(-beta*d) -> [0,1])
        reliability_thr: passed to metric_fn
        sim_thresh: metric similarity above which to apply decay
        score_thresh: minimum score to keep a box
        decay: "linear" or "gaussian"

    Returns:
        keep_indices, decayed_scores
    """
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device), scores

    if metric_fn is None or boxes.numel() <= 1:
        return torch.arange(len(boxes), device=boxes.device), scores

    xn = (boxes[:, 0] + boxes[:, 2]) / 2.0
    yn = (boxes[:, 1] + boxes[:, 3]) / 2.0
    wn = (boxes[:, 2] - boxes[:, 0]).clamp(min=1.0)
    hn = (boxes[:, 3] - boxes[:, 1]).clamp(min=1.0)
    sim = metric_fn(xn, yn, wn, hn, xn, yn, wn, hn,
                    reliability_thr=reliability_thr)

    N = len(boxes)
    scores_decayed = scores.clone()
    order = scores_decayed.argsort(descending=True)

    for idx_pos in range(N):
        i = order[idx_pos].item()
        if scores_decayed[i] < score_thresh:
            continue
        # Decay scores of all remaining lower-ranked boxes
        for pos_j in range(idx_pos + 1, N):
            j = order[pos_j].item()
            if scores_decayed[j] < score_thresh:
                continue
            sij = sim[i, j].item()
            if sij > sim_thresh:
                if decay == "linear":
                    scores_decayed[j] *= (1.0 - sij)
                else:  # gaussian
                    scores_decayed[j] *= torch.exp(-sij * sij / 0.5)

    keep = scores_decayed >= score_thresh
    return torch.where(keep)[0], scores_decayed[keep]


# =============================================================================
# Main builder
# =============================================================================
def build_model(
    metric_fn: Optional[Callable] = None,
    placement: str = "everywhere",
    reliability_thr: float = 16.0,
    num_classes: int = NUM_CLASSES,
    channels_last: bool = False,
    saalw_rpn_cfg: Optional[dict] = None,
    box_loss_type: str = "metric",
    box_loss_warmup_epochs: int = BOX_LOSS_WARMUP_EPOCHS,
    use_quality_score: bool = False,
    quality_loss_weight: float = 0.0,
    use_quality_focal: bool = False,
    quality_focal_beta: float = 2.0,
    use_rank_sort: bool = False,
    rank_sort_delta: float = 0.5,
    use_double_head: bool = False,
    double_head_reg_roi_scale: float = 1.3,
    double_head_num_convs: int = 4,
    cbl_refine_steps: int = 0,
    cbl_refine_blend: float = 1.0,
    cbl_refine_score_threshold: float = 0.0,
    cbl_refine_train_weight: float = 0.0,
    cbl_refine_separate_head: bool = False,
    cbl_alpha: float = CBL_ALPHA,
    cbl_num_bins: int = CBL_NUM_BINS,
    cbl_grid_beta: float = CBL_GRID_BETA,
    cbl_um_weight: float = CBL_UM_WEIGHT,
) -> nn.Module:
    """Build a Faster R-CNN with metric at given placements.

    Args:
        metric_fn: callable; if None → standard torchvision
        placement: one of:
            - "everywhere":       baseline (metric_fn ignored)
            - "la":               metric in RPN label assignment (hierarchical)
            - "la_loss":          metric in RPN LA + RoI box loss
            - "la_loss_nms":      LA + loss + NMS (hard metric-NMS)
            - "la_loss_soft_nms": LA + loss + Soft-NMS (ALW score decay)
            - "saalw_assigner":   SAALWAssigner (threshold-based) + box loss
        reliability_thr: passed to metric
        saalw_rpn_cfg: dict of SAALWAssigner params
        box_loss_type: "metric", "smooth_l1", "side_smooth_l1", "ciou",
            "diou", or "cbl"
        box_loss_warmup_epochs: number of warmup epochs with pure metric loss
        use_quality_score: add an auxiliary RoI localization-quality head
        quality_loss_weight: BCE weight for IoU quality target on positive RoIs
        use_quality_focal: replace RoI softmax CE with joint class-IoU QFL
        quality_focal_beta: QFL modulating exponent
        use_rank_sort: replace RoI softmax CE with sampled Rank & Sort loss
        rank_sort_delta: smoothing width for Rank & Sort score comparisons
        use_double_head: use FC classification and convolutional CBL regression
        double_head_reg_roi_scale: proposal enlargement for regression features
        double_head_num_convs: residual bottlenecks in the regression branch
        cbl_refine_steps: inference-only repeated CBL box-regression passes
        cbl_refine_blend: fraction of each predicted refinement update
        cbl_refine_score_threshold: preserve boxes below this class score
        cbl_refine_train_weight: auxiliary shared-head second-pass CBL weight
        cbl_refine_separate_head: specialize a cloned head for the second pass
        cbl_alpha: normalized delta range for confidence-driven localization
        cbl_num_bins: number of distribution logits per box coordinate
        cbl_grid_beta: interval-nonuniform grid density around zero
        cbl_um_weight: entropy-matching uncertainty loss weight
        channels_last: if True, convert backbone to channels_last memory format
    """
    if placement not in ("everywhere", "la", "la_loss", "la_loss_nms", "la_loss_soft_nms", "saalw_assigner"):
        raise ValueError(f"Unknown placement: {placement}")

    base = fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT,
        min_size=MIN_SIZE, max_size=MAX_SIZE,
        box_detections_per_img=BOX_DETECTIONS_PER_IMG,
        box_nms_thresh=NMS_THRESH_TEST,
        box_score_thresh=SCORE_THRESH_TRAIN,
        rpn_pre_nms_top_n_train=RPN_NUM_PROPOSALS_TRAIN,
        rpn_post_nms_top_n_train=RPN_NUM_PROPOSALS_TRAIN,
        rpn_pre_nms_top_n_test=RPN_NUM_PROPOSALS_TEST,
        rpn_post_nms_top_n_test=RPN_NUM_PROPOSALS_TEST,
        rpn_batch_size_per_image=256,
        rpn_positive_fraction=0.5,
    )

    # Speed optimization: channels_last memory format for backbone
    if channels_last:
        # Prefer backbone.conv_body if available, else fallback to backbone
        backbone = getattr(base, "backbone", None)
        target = getattr(backbone, "conv_body", backbone) if backbone is not None else None
        if target is not None:
            target.to(memory_format=torch.channels_last)
            print(f"  [speed] backbone converted to channels_last memory format")

    anchor_gen = AnchorGenerator(
        sizes=((4, 8, 16), (16, 32, 64), (64, 128, 256),
               (128, 256, 512), (256, 512, 1024)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5)

    use_metric_rpn  = placement in ("la", "la_loss", "la_loss_nms", "la_loss_soft_nms") and metric_fn is not None
    use_metric_loss = placement in ("la_loss", "la_loss_nms", "la_loss_soft_nms") and metric_fn is not None
    use_metric_nms  = placement in ("la_loss_nms",) and metric_fn is not None
    use_soft_nms    = placement in ("la_loss_soft_nms",) and metric_fn is not None
    use_saalw_rpn   = placement == "saalw_assigner"

    if use_saalw_rpn and metric_fn is not None:
        from .assigner import SAALWAssigner
        cfg = saalw_rpn_cfg or {}
        assigner = SAALWAssigner(
            metric_fn=metric_fn,
            pos_sim_thr=cfg.get("pos_sim_thr", 0.45),
            neg_sim_thr=cfg.get("neg_sim_thr", 0.20),
            topk_fallback=cfg.get("topk_fallback", 6),
            dynamic_thr=cfg.get("dynamic_thr", True),
            reliability_thr=reliability_thr,
        )
        print(f"  [SAALW] pos_thr={assigner.pos_sim_thr}, neg_thr={assigner.neg_sim_thr}, "
              f"topk={assigner.topk_fallback}, dynamic={assigner.dynamic_thr}")
        base.rpn = SAALWRPN(
            anchor_generator=anchor_gen,
            head=RPNHead(base.backbone.out_channels,
                         anchor_gen.num_anchors_per_location()[0]),
            fg_iou_thresh=RPN_FG_IOU, bg_iou_thresh=RPN_BG_IOU,
            batch_size_per_image=256, positive_fraction=0.5,
            pre_nms_top_n={"training": RPN_NUM_PROPOSALS_TRAIN,
                           "testing": RPN_NUM_PROPOSALS_TEST},
            post_nms_top_n={"training": RPN_NUM_PROPOSALS_TRAIN,
                            "testing": RPN_NUM_PROPOSALS_TEST},
            nms_thresh=RPN_NMS_THRESH,
            saalw_assigner=assigner,
        )
    elif use_metric_rpn:
        base.rpn = MetricRPN(
            anchor_generator=anchor_gen,
            head=RPNHead(base.backbone.out_channels,
                         anchor_gen.num_anchors_per_location()[0]),
            fg_iou_thresh=RPN_FG_IOU, bg_iou_thresh=RPN_BG_IOU,
            batch_size_per_image=256, positive_fraction=0.5,
            pre_nms_top_n={"training": RPN_NUM_PROPOSALS_TRAIN,
                           "testing": RPN_NUM_PROPOSALS_TEST},
            post_nms_top_n={"training": RPN_NUM_PROPOSALS_TRAIN,
                            "testing": RPN_NUM_PROPOSALS_TEST},
            nms_thresh=RPN_NMS_THRESH,
            metric_fn=metric_fn,
            reliability_thr=reliability_thr,
        )
    else:
        base.rpn = RegionProposalNetwork(
            anchor_generator=anchor_gen,
            head=RPNHead(base.backbone.out_channels,
                         anchor_gen.num_anchors_per_location()[0]),
            fg_iou_thresh=RPN_FG_IOU, bg_iou_thresh=RPN_BG_IOU,
            batch_size_per_image=256, positive_fraction=0.5,
            pre_nms_top_n={"training": RPN_NUM_PROPOSALS_TRAIN,
                           "testing": RPN_NUM_PROPOSALS_TEST},
            post_nms_top_n={"training": RPN_NUM_PROPOSALS_TRAIN,
                            "testing": RPN_NUM_PROPOSALS_TEST},
            nms_thresh=RPN_NMS_THRESH,
        )

    base.roi_heads.fg_iou_thresh = ROI_FG_IOU_THRESH
    base.roi_heads.bg_iou_thresh = ROI_BG_IOU_THRESH
    base.roi_heads.batch_size_per_image = 256
    base.roi_heads.positive_fraction = 0.5
    in_feat = base.roi_heads.box_predictor.cls_score.in_features
    if use_quality_focal and box_loss_type != "cbl":
        raise ValueError("The bounded QFL experiment requires CBL localization")
    if use_rank_sort and box_loss_type != "cbl":
        raise ValueError("The bounded Rank & Sort experiment requires CBL localization")
    if use_double_head and box_loss_type != "cbl":
        raise ValueError("The Double-Head experiment requires CBL localization")
    if cbl_refine_steps < 0:
        raise ValueError("CBL refine steps cannot be negative")
    if cbl_refine_blend <= 0:
        raise ValueError("CBL refine blend must be positive")
    if not 0 <= cbl_refine_score_threshold <= 1:
        raise ValueError("CBL refine score threshold must be in [0, 1]")
    if cbl_refine_train_weight < 0:
        raise ValueError("CBL refine training weight cannot be negative")
    if cbl_refine_separate_head and (
            cbl_refine_train_weight <= 0 or cbl_refine_steps != 1):
        raise ValueError(
            "Stage-specific CBL refinement requires positive training weight "
            "and exactly one inference step")
    if cbl_refine_steps > 0 and box_loss_type != "cbl":
        raise ValueError("Iterative CBL refinement requires CBL localization")
    if use_quality_focal and use_rank_sort:
        raise ValueError("QFL and Rank & Sort are mutually exclusive")
    if use_double_head and (use_quality_focal or use_rank_sort):
        raise ValueError(
            "Double-Head must be evaluated without QFL or Rank & Sort")
    if cbl_refine_steps > 0 and (
            use_double_head or use_quality_focal or use_rank_sort):
        raise ValueError(
            "Iterative CBL refinement requires the standard softmax CBL head")
    if cbl_refine_train_weight > 0 and (
            box_loss_type != "cbl" or use_double_head
            or use_quality_focal or use_rank_sort):
        raise ValueError(
            "Iterative CBL training requires the standard softmax CBL head")
    if use_quality_focal and use_quality_score:
        raise ValueError("QFL cannot be combined with the standalone quality head")
    if use_rank_sort and use_quality_score:
        raise ValueError(
            "Rank & Sort cannot be combined with the standalone quality head"
        )
    if box_loss_type == "cbl" and use_quality_score:
        raise ValueError("CBL and the standalone quality head cannot be combined")
    if box_loss_type == "cbl":
        if use_double_head:
            base.roi_heads.box_predictor = DoubleHeadCBLPredictor(
                in_feat,
                base.backbone.out_channels,
                num_classes + 1,
                alpha=cbl_alpha,
                num_bins=cbl_num_bins,
                grid_beta=cbl_grid_beta,
                num_convs=double_head_num_convs,
            )
        else:
            base.roi_heads.box_predictor = CBLFastRCNNPredictor(
                in_feat, num_classes + 1,
                alpha=cbl_alpha,
                num_bins=cbl_num_bins,
                grid_beta=cbl_grid_beta,
            )
    elif use_quality_score:
        base.roi_heads.box_predictor = QualityFastRCNNPredictor(in_feat, num_classes + 1)
    else:
        base.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes + 1)

    if cbl_refine_separate_head:
        base.roi_heads.refine_box_head = copy.deepcopy(
            base.roi_heads.box_head)
        base.roi_heads.refine_box_predictor = copy.deepcopy(
            base.roi_heads.box_predictor)
        base.roi_heads.refine_box_predictor.cls_score.requires_grad_(False)

    # ── Replace box regression loss with metric distance ──────────────
    if (use_metric_loss or use_quality_focal or use_rank_sort or use_double_head or
            (use_quality_score and quality_loss_weight > 0)):
        _wrap_roi_forward_for_metric_loss(
            base.roi_heads, metric_fn, reliability_thr,
            metric_loss_weight=1.0,
            box_loss_type=box_loss_type,
            box_loss_warmup_epochs=box_loss_warmup_epochs,
            quality_loss_weight=quality_loss_weight if use_quality_score else 0.0,
            use_quality_focal=use_quality_focal,
            quality_focal_beta=quality_focal_beta,
            use_rank_sort=use_rank_sort,
            rank_sort_delta=rank_sort_delta,
            use_double_head=use_double_head,
            double_head_reg_roi_scale=double_head_reg_roi_scale,
            cbl_refine_steps=cbl_refine_steps,
            cbl_refine_blend=cbl_refine_blend,
            cbl_refine_score_threshold=cbl_refine_score_threshold,
            cbl_refine_train_weight=cbl_refine_train_weight,
            cbl_refine_separate_head=cbl_refine_separate_head,
            cbl_um_weight=cbl_um_weight)
        print(f"  [loss] fastrcnn_loss replaced, type={box_loss_type}, "
              f"warmup_epochs={box_loss_warmup_epochs}")
        if box_loss_type == "cbl":
            print(f"  [CBL] alpha={cbl_alpha:g}, bins={cbl_num_bins}, "
                  f"grid_beta={cbl_grid_beta:g}, um_weight={cbl_um_weight:g}")
        if use_quality_score:
            print(f"  [quality] score=head enabled, loss_weight={quality_loss_weight}")
        if use_quality_focal:
            print(f"  [QFL] joint class-IoU score enabled, beta={quality_focal_beta:g}")
        if use_rank_sort:
            print(f"  [RankSort] sampled RoI ranking enabled, delta={rank_sort_delta:g}")
        if use_double_head:
            print(
                "  [DoubleHead] FC classification + convolutional CBL "
                f"regression, roi_scale={double_head_reg_roi_scale:g}, "
                f"bottlenecks={double_head_num_convs}"
            )
        if cbl_refine_steps > 0:
            print(
                f"  [CBL refine] inference passes={cbl_refine_steps}, "
                f"blend={cbl_refine_blend:g}, "
                f"score_threshold={cbl_refine_score_threshold:g}"
            )
        if cbl_refine_train_weight > 0:
            print(
                "  [CBL refine train] "
                f"{'stage-specific' if cbl_refine_separate_head else 'shared-head'} "
                f"second pass, loss_weight={cbl_refine_train_weight:g}"
            )

    # ── Wrap forward to optionally use metric NMS in inference ──────────
    if use_metric_nms:
        base.roi_heads._use_metric_nms = True
        base.roi_heads._metric_fn = metric_fn
        base.roi_heads._reliability_thr = reliability_thr
        _wrap_postprocess_for_metric_nms(base.roi_heads, metric_fn, reliability_thr)

    if use_soft_nms:
        base.roi_heads._use_soft_nms = True
        base.roi_heads._metric_fn = metric_fn
        base.roi_heads._reliability_thr = reliability_thr
        _wrap_postprocess_for_soft_metric_nms(base.roi_heads, metric_fn, reliability_thr)

    return base


def _wrap_postprocess_for_metric_nms(roi_heads, metric_fn, reliability_thr):
    """Override RoIHeads.postprocess_detections to use metric NMS.

    torchvision 0.26 signature:
        postprocess_detections(self, class_logits, box_regression,
                               proposals, image_shapes)
        -> (all_boxes, all_scores, all_labels)
    where each is List[Tensor].

    We wrap to apply metric-NMS after standard postprocessing.
    """
    original_postprocess = roi_heads.postprocess_detections

    def postprocess_with_metric_nms(
        self, class_logits, box_regression, proposals, image_shapes
    ):
        all_boxes, all_scores, all_labels = original_postprocess(
            class_logits, box_regression, proposals, image_shapes)

        new_boxes, new_scores, new_labels = [], [], []
        for boxes, scores, labels in zip(all_boxes, all_scores, all_labels):
            if boxes.numel() == 0:
                new_boxes.append(boxes)
                new_scores.append(scores)
                new_labels.append(labels)
                continue
            keep_boxes = []
            keep_scores = []
            keep_labels = []
            for cls in labels.unique():
                mask = labels == cls
                kb = metric_nms(
                    boxes[mask], scores[mask],
                    metric_fn=metric_fn, reliability_thr=reliability_thr,
                )
                keep_boxes.append(boxes[mask][kb])
                keep_scores.append(scores[mask][kb])
                keep_labels.append(labels[mask][kb])
            if keep_boxes:
                new_boxes.append(torch.cat(keep_boxes))
                new_scores.append(torch.cat(keep_scores))
                new_labels.append(torch.cat(keep_labels))
            else:
                new_boxes.append(torch.zeros(0, 4, device=boxes.device))
                new_scores.append(torch.zeros(0, device=boxes.device))
                new_labels.append(torch.zeros(0, dtype=torch.int64, device=boxes.device))
        return new_boxes, new_scores, new_labels

    import types
    roi_heads.postprocess_detections = types.MethodType(
        postprocess_with_metric_nms, roi_heads)


def _wrap_postprocess_for_soft_metric_nms(roi_heads, metric_fn, reliability_thr):
    """Override RoIHeads.postprocess_detections to use Soft metric-NMS.

    Instead of hard suppressing boxes, decay their scores using ALW similarity.
    """
    original_postprocess = roi_heads.postprocess_detections

    def postprocess_with_soft_metric_nms(
        self, class_logits, box_regression, proposals, image_shapes
    ):
        all_boxes, all_scores, all_labels = original_postprocess(
            class_logits, box_regression, proposals, image_shapes)

        new_boxes, new_scores, new_labels = [], [], []
        for boxes, scores, labels in zip(all_boxes, all_scores, all_labels):
            if boxes.numel() == 0:
                new_boxes.append(boxes)
                new_scores.append(scores)
                new_labels.append(labels)
                continue
            keep_boxes = []
            keep_scores = []
            keep_labels = []
            for cls in labels.unique():
                mask = labels == cls
                ki, ks = soft_metric_nms(
                    boxes[mask], scores[mask],
                    metric_fn=metric_fn, reliability_thr=reliability_thr,
                )
                keep_boxes.append(boxes[mask][ki])
                keep_scores.append(ks)
                keep_labels.append(labels[mask][ki])
            if keep_boxes:
                new_boxes.append(torch.cat(keep_boxes))
                new_scores.append(torch.cat(keep_scores))
                new_labels.append(torch.cat(keep_labels))
            else:
                new_boxes.append(torch.zeros(0, 4, device=boxes.device))
                new_scores.append(torch.zeros(0, device=boxes.device))
                new_labels.append(torch.zeros(0, dtype=torch.int64, device=boxes.device))
        return new_boxes, new_scores, new_labels

    import types
    roi_heads.postprocess_detections = types.MethodType(
        postprocess_with_soft_metric_nms, roi_heads)
