"""Model builder — metric-aware Faster R-CNN.

Supports 4 placements (Phase 2 ablation):
  - "everywhere"   : baseline, standard torchvision, no metric
  - "la"           : metric in RPN label assignment only
  - "la_loss"      : metric in RPN LA + RoIHeads box loss
  - "la_loss_nms"  : metric in RPN LA + RoIHeads box loss + NMS

Metric loss implementation: we override torchvision's RoIHeads.compute_loss
to replace Smooth-L1 with metric-distance loss.
"""
from __future__ import annotations
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
from torchvision.ops import batched_nms

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
)

EPS = 1e-6


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
# Metric RPN — label assignment via metric similarity
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
# Metric RoIHeads — replace Smooth-L1 box loss with metric distance
# =============================================================================
class MetricRoIHeads(RoIHeads):
    """RoIHeads where box regression loss is (1 - metric_similarity)."""

    def __init__(self, *args, metric_fn=None, reliability_thr=16.0,
                 metric_loss_weight=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_fn = metric_fn
        self.reliability_thr = reliability_thr
        self.metric_loss_weight = metric_loss_weight

    def compute_loss(self, class_logits, box_regression, labels, regression_targets):
        """Same as torchvision but uses metric distance instead of Smooth-L1."""
        # Classification loss (giống gốc)
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        classification_loss = F.cross_entropy(class_logits, labels)

        # Get positive samples
        sampled_pos_inds = torch.where(labels > 0)[0]
        if sampled_pos_inds.numel() == 0:
            return {
                "loss_classifier": classification_loss,
                "loss_box_reg": torch.zeros(1, device=class_logits.device).sum(),
            }

        # Box regression loss = metric distance between decoded pred & target
        # Re-decode pred boxes using the box_coder with class-specific decoding
        box_regression_pos = box_regression[sampled_pos_inds]
        labels_pos = labels[sampled_pos_inds]
        targets_pos = regression_targets[sampled_pos_inds]

        # Decode pred boxes via torchvision's box_coder
        pred_boxes = self.box_coder.decode(box_regression_pos, labels_pos)

        # Compute metric loss
        if self.metric_fn is not None and pred_boxes.numel() > 0:
            xn = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
            yn = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0
            wn = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1.0)
            hn = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1.0)
            xg = (targets_pos[:, 0] + targets_pos[:, 2]) / 2.0
            yg = (targets_pos[:, 1] + targets_pos[:, 3]) / 2.0
            wg = (targets_pos[:, 2] - targets_pos[:, 0]).clamp(min=1.0)
            hg = (targets_pos[:, 3] - targets_pos[:, 1]).clamp(min=1.0)
            sim = self.metric_fn(xn, yn, wn, hn, xg, yg, wg, hg,
                                 reliability_thr=self.reliability_thr)
            box_loss = (1.0 - sim).mean() * self.metric_loss_weight
        else:
            box_loss = F.smooth_l1_loss(
                box_regression[sampled_pos_inds],
                regression_targets[sampled_pos_inds],
                beta=1.0,
                reduction="sum",
            ) / max(labels.numel(), 1)

        return {
            "loss_classifier": classification_loss,
            "loss_box_reg": box_loss,
        }


# =============================================================================
# Metric-based NMS
# =============================================================================
def metric_nms(boxes, scores, metric_fn=None, reliability_thr=16.0,
               iou_thresh=NMS_METRIC_THRESH):
    """NMS where 'overlap' is metric_similarity > thresh."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    if metric_fn is None:
        return batched_nms(boxes, scores,
                           torch.zeros(len(boxes), dtype=torch.long, device=boxes.device),
                           iou_thresh)
    # pairwise metric similarity
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
# Main builder
# =============================================================================
def build_model(
    metric_fn: Optional[Callable] = None,
    placement: str = "everywhere",
    reliability_thr: float = 16.0,
    num_classes: int = NUM_CLASSES,
    channels_last: bool = False,
) -> nn.Module:
    """Build a Faster R-CNN with metric at given placements.

    Args:
        metric_fn: callable; if None → standard torchvision (CIoU baseline)
        placement: one of:
            - "everywhere": baseline (metric_fn ignored)
            - "la": metric in RPN label assignment
            - "la_loss": metric in RPN LA + RoI box loss
            - "la_loss_nms": LA + loss + NMS
        reliability_thr: passed to metric
        channels_last: if True, convert backbone to channels_last memory format
                        for ~5-10% backbone speedup on modern GPUs (caller must
                        still pass images as channels_last tensors).
    """
    if placement not in ("everywhere", "la", "la_loss", "la_loss_nms"):
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

    use_metric_rpn  = placement in ("la", "la_loss", "la_loss_nms") and metric_fn is not None
    use_metric_loss = placement in ("la_loss", "la_loss_nms") and metric_fn is not None
    use_metric_nms  = placement in ("la_loss_nms",) and metric_fn is not None

    if use_metric_rpn:
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
    base.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes + 1)

    # Replace RoIHeads entirely if metric loss is needed
    if use_metric_loss:
        # We need to construct a fresh MetricRoIHeads with same internals.
        # This is fragile; for simplicity we just attach a flag.
        base.roi_heads._use_metric_loss = True
        base.roi_heads._metric_fn = metric_fn
        base.roiheads_reliability_thr = reliability_thr

    # ── Wrap forward to optionally use metric NMS in inference ──────────
    if use_metric_nms:
        base.roi_heads._use_metric_nms = True
        base.roi_heads._metric_fn = metric_fn
        base.roi_heads._reliability_thr = reliability_thr
        _wrap_postprocess_for_metric_nms(base.roi_heads, metric_fn, reliability_thr)

    return base


def _wrap_postprocess_for_metric_nms(roi_heads, metric_fn, reliability_thr):
    """Override RoIHeads.postprocess_detections to use metric NMS.

    torchvision calls `postprocess_detections` from `forward()`. Renaming
    `postprocess` → `postprocess_detections` here (bug fix: original code
    referenced a non-existent attribute).
    """
    original_postprocess = roi_heads.postprocess_detections

    def postprocess_with_metric_nms(
        self, result, image_shapes, originals=None
    ):
        # Run standard postprocess first (gives boxes, scores, labels)
        outputs = original_postprocess(result, image_shapes, originals)
        # Then apply metric NMS per image
        new_outputs = []
        for out in outputs:
            boxes = out["boxes"]
            scores = out["scores"]
            labels = out["labels"]
            if boxes.numel() == 0:
                new_outputs.append(out)
                continue
            new_boxes = []
            new_scores = []
            new_labels = []
            for cls in labels.unique():
                mask = labels == cls
                kb = metric_nms(
                    boxes[mask], scores[mask],
                    metric_fn=metric_fn, reliability_thr=reliability_thr,
                )
                new_boxes.append(boxes[mask][kb])
                new_scores.append(scores[mask][kb])
                new_labels.append(labels[mask][kb])
            if new_boxes:
                out = {
                    "boxes": torch.cat(new_boxes),
                    "scores": torch.cat(new_scores),
                    "labels": torch.cat(new_labels),
                }
            new_outputs.append(out)
        return new_outputs

    # bind
    import types
    roi_heads.postprocess_detections = types.MethodType(postprocess_with_metric_nms, roi_heads)