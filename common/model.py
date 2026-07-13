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
from torchvision.ops import batched_nms, complete_box_iou_loss, distance_box_iou_loss

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
def _metric_box_loss(class_logits, box_regression, labels, regression_targets,
                     box_coder, metric_fn, reliability_thr, metric_loss_weight,
                     proposals, current_epoch=1, box_loss_type=None):
    """Replacement for torchvision's fastrcnn_loss with multi-loss dispatch.

    Args:
        current_epoch: used for warmup (pure metric loss first N epochs, then ramp)
        box_loss_type: "metric", "smooth_l1", "ciou", or "diou" (defaults to config)
    """
    if box_loss_type is None:
        box_loss_type = BOX_LOSS_TYPE
    labels = torch.cat(labels, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)
    classification_loss = F.cross_entropy(class_logits, labels)

    sampled_pos_inds = torch.where(labels > 0)[0]
    if sampled_pos_inds.numel() == 0:
        return classification_loss, torch.zeros(1, device=class_logits.device).sum()

    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, num_classes, 4)
    K = len(sampled_pos_inds)
    box_regression_pos = box_regression[sampled_pos_inds]
    labels_pos = labels[sampled_pos_inds]
    targets_deltas = regression_targets[sampled_pos_inds]

    proposals_flat = torch.cat(proposals, dim=0)
    proposals_pos = proposals_flat[sampled_pos_inds]

    # ── Determine loss type ──
    loss_type = box_loss_type

    # ── Warmup: pure metric loss for first warmup_epochs ──
    if loss_type != "metric" and current_epoch <= BOX_LOSS_WARMUP_EPOCHS:
        loss_type = "metric"

    # ── Compute primary box loss ──
    if loss_type == "metric":
        # Current Gaussian similarity loss
        box_reg_flat = box_regression_pos.reshape(K, num_classes * 4)
        decoded = box_coder.decode(box_reg_flat, [proposals_pos])
        pred_boxes = decoded[torch.arange(K, device=decoded.device), labels_pos]
        decoded_gt = box_coder.decode(targets_deltas, [proposals_pos])
        gt_boxes = decoded_gt[:, 0, :]

        xn = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
        yn = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0
        wn = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1.0)
        hn = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1.0)
        xg = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
        yg = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
        wg = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1.0)
        hg = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1.0)

        sim = metric_fn(xn, yn, wn, hn, xg, yg, wg, hg,
                         reliability_thr=reliability_thr)
        box_loss = (1.0 - sim).mean() * metric_loss_weight

    elif loss_type == "smooth_l1":
        # Standard Smooth-L1 on delta space (mirrors RFLA's AP75=18.8)
        # Select the regression output for the positive class
        box_reg_pos_per_class = box_regression_pos[
            torch.arange(K, device=box_regression_pos.device), labels_pos]  # (K, 4)
        box_loss = F.smooth_l1_loss(
            box_reg_pos_per_class, targets_deltas, beta=1.0)
        # Add auxiliary metric loss (small weight) to preserve micro stability
        box_reg_flat = box_regression_pos.reshape(K, num_classes * 4)
        decoded = box_coder.decode(box_reg_flat, [proposals_pos])
        pred_boxes = decoded[torch.arange(K, device=decoded.device), labels_pos]
        decoded_gt = box_coder.decode(targets_deltas, [proposals_pos])
        gt_boxes = decoded_gt[:, 0, :]
        xn = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2.0
        yn = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2.0
        wn = (pred_boxes[:, 2] - pred_boxes[:, 0]).clamp(min=1.0)
        hn = (pred_boxes[:, 3] - pred_boxes[:, 1]).clamp(min=1.0)
        xg = (gt_boxes[:, 0] + gt_boxes[:, 2]) / 2.0
        yg = (gt_boxes[:, 1] + gt_boxes[:, 3]) / 2.0
        wg = (gt_boxes[:, 2] - gt_boxes[:, 0]).clamp(min=1.0)
        hg = (gt_boxes[:, 3] - gt_boxes[:, 1]).clamp(min=1.0)
        sim = metric_fn(xn, yn, wn, hn, xg, yg, wg, hg,
                         reliability_thr=reliability_thr)
        metric_aux = (1.0 - sim).mean()
        box_loss = box_loss + BOX_LOSS_METRIC_WEIGHT * metric_aux

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

    return classification_loss, box_loss


def _wrap_roi_forward_for_metric_loss(roi_heads, metric_fn, reliability_thr,
                                       metric_loss_weight=1.0,
                                       box_loss_type="metric"):
    """Monkey-patch RoIHeads.forward to replace fastrcnn_loss with metric loss."""
    original_forward = roi_heads.forward
    # Store on roi_heads for access during training
    roi_heads._box_loss_type = box_loss_type

    def patched_forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None and self.training:
            proposals_sampled, matched_idxs, labels, regression_targets = \
                self.select_training_samples(proposals, targets)
        else:
            labels = None; regression_targets = None; matched_idxs = None
            proposals_sampled = proposals

        box_features = self.box_roi_pool(features, proposals_sampled, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = []
        losses = {}
        if self.training:
            current_epoch = getattr(self, '_current_epoch', 1)
            box_loss_type = getattr(self, '_box_loss_type', BOX_LOSS_TYPE)
            loss_classifier, loss_box_reg = _metric_box_loss(
                class_logits, box_regression, labels, regression_targets,
                self.box_coder, metric_fn, reliability_thr, metric_loss_weight,
                proposals_sampled, current_epoch=current_epoch,
                box_loss_type=box_loss_type)
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
        else:
            boxes, scores, labels_out = self.postprocess_detections(
                class_logits, box_regression, proposals, image_shapes)
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
        box_loss_type: "metric", "smooth_l1", "ciou", or "diou"
        box_loss_warmup_epochs: number of warmup epochs with pure metric loss
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
    base.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes + 1)

    # ── Replace box regression loss with metric distance ──────────────
    if use_metric_loss:
        _wrap_roi_forward_for_metric_loss(
            base.roi_heads, metric_fn, reliability_thr,
            metric_loss_weight=1.0,
            box_loss_type=box_loss_type)
        print(f"  [loss] fastrcnn_loss replaced, type={box_loss_type}")

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