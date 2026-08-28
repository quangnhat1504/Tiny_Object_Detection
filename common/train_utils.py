"""Training utilities — shared across all experiments."""
from __future__ import annotations
import math
from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from .config import (
    DEVICE, EMPTY_CACHE_EVERY, USE_EMA, EMA_DECAY,
)


def _backward_with_pcgrad(
    detector_loss: torch.Tensor,
    auxiliary_loss: torch.Tensor,
    shared_parameters,
    scaler,
) -> dict[str, float]:
    """Backpropagate while removing only conflicting auxiliary components."""
    return _backward_with_disjoint_pcgrad(
        detector_loss,
        (("auxiliary", auxiliary_loss, shared_parameters),),
        scaler,
    )["auxiliary"]


def _backward_with_disjoint_pcgrad(
    detector_loss: torch.Tensor,
    auxiliary_specs,
    scaler,
) -> dict[str, dict[str, float]]:
    """Project multiple auxiliaries independently on disjoint parameter scopes."""
    normalized_specs = []
    parameter_ids = set()
    all_parameters = []
    for name, auxiliary_loss, parameters in auxiliary_specs:
        parameters = tuple(
            parameter for parameter in parameters if parameter.requires_grad)
        if not parameters:
            raise ValueError(
                f"PCGrad scope {name!r} has no trainable parameters")
        overlap = parameter_ids.intersection(map(id, parameters))
        if overlap:
            raise ValueError("PCGrad auxiliary parameter scopes must be disjoint")
        parameter_ids.update(map(id, parameters))
        all_parameters.extend(parameters)
        normalized_specs.append((name, auxiliary_loss, parameters))
    if not normalized_specs:
        raise ValueError("PCGrad requires at least one auxiliary loss")

    detector_gradients = torch.autograd.grad(
        scaler.scale(detector_loss),
        tuple(all_parameters),
        retain_graph=True,
        allow_unused=True,
    )
    auxiliary_gradients = []
    for _, auxiliary_loss, parameters in normalized_specs:
        auxiliary_gradients.append(torch.autograd.grad(
            scaler.scale(auxiliary_loss),
            parameters,
            retain_graph=True,
            allow_unused=True,
        ))
    scaler.scale(
        detector_loss + sum(
            auxiliary_loss
            for _, auxiliary_loss, _ in normalized_specs
        )
    ).backward()

    results = {}
    cursor = 0
    reference = detector_loss.detach().float()
    for (name, _, parameters), gradients in zip(
        normalized_specs, auxiliary_gradients
    ):
        scope_detector_gradients = detector_gradients[
            cursor:cursor + len(parameters)]
        cursor += len(parameters)
        dot = reference.new_zeros(())
        detector_norm_sq = reference.new_zeros(())
        auxiliary_norm_sq = reference.new_zeros(())
        for detector_gradient, auxiliary_gradient in zip(
            scope_detector_gradients, gradients
        ):
            if detector_gradient is not None:
                detector_norm_sq += (
                    detector_gradient.detach().float().square().sum())
            if auxiliary_gradient is not None:
                auxiliary_norm_sq += (
                    auxiliary_gradient.detach().float().square().sum())
            if detector_gradient is not None and auxiliary_gradient is not None:
                dot += (
                    detector_gradient.detach().float()
                    * auxiliary_gradient.detach().float()
                ).sum()

        denominator = detector_norm_sq.clamp_min(
            torch.finfo(torch.float32).eps)
        projection_coefficient = (
            torch.minimum(dot, dot.new_zeros(())) / denominator)
        for parameter, detector_gradient, auxiliary_gradient in zip(
            parameters, scope_detector_gradients, gradients
        ):
            if detector_gradient is None and auxiliary_gradient is None:
                parameter.grad = None
                continue
            detector_component = (
                torch.zeros_like(parameter)
                if detector_gradient is None else detector_gradient)
            auxiliary_component = (
                torch.zeros_like(parameter)
                if auxiliary_gradient is None else auxiliary_gradient)
            parameter.grad = (
                detector_component
                + auxiliary_component
                - projection_coefficient.to(detector_component.dtype)
                * detector_component
            ).detach()

        cosine_denominator = (
            detector_norm_sq.sqrt() * auxiliary_norm_sq.sqrt()
        ).clamp_min(torch.finfo(torch.float32).eps)
        results[name] = {
            "conflict": float((dot < 0).item()),
            "cosine": float((dot / cosine_denominator).item()),
            "auxiliary_norm_ratio": float((
                auxiliary_norm_sq.sqrt()
                / detector_norm_sq.sqrt().clamp_min(
                    torch.finfo(torch.float32).eps)
            ).item()),
        }
    return results


# =============================================================================
# Model EMA — exponential moving average of weights
# =============================================================================
class ModelEMA:
    """Maintain shadow copy of model weights via EMA on model's device."""

    def __init__(self, model: nn.Module, decay: float = EMA_DECAY):
        self.decay = decay
        # Keep shadow on the SAME device as model to avoid per-step CPU sync.
        self.shadow = deepcopy(model)
        try:
            model_device = next(model.parameters()).device
            self.shadow.to(model_device)
        except StopIteration:
            pass
        self.shadow.eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        d = self.decay
        one_minus_d = 1.0 - d
        # All ops stay on model's device (no CPU sync).
        for s, m in zip(self.shadow.parameters(), model.parameters()):
            s.data.mul_(d).add_(m.data, alpha=one_minus_d)
        for sb, mb in zip(self.shadow.buffers(), model.buffers()):
            sb.data.mul_(d).add_(mb.data, alpha=one_minus_d)

    def get_model(self) -> nn.Module:
        return self.shadow

    def state_dict(self) -> dict:
        return self.shadow.state_dict()

    def load_state_dict(self, sd):
        self.shadow.load_state_dict(sd)


# =============================================================================
# Warmup + Cosine LR scheduler
# =============================================================================
class WarmupCosineLR:
    """Linear warmup → cosine decay."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 base_lr: float, warmup_start_lr: float):
        self.opt = optimizer
        self.warmup_ep = max(1, warmup_epochs)
        self.total_ep = max(2, total_epochs)
        self.base_lr = base_lr
        self.warm_lr0 = warmup_start_lr
        self.cur_epoch = 0

    def step_epoch(self):
        self.cur_epoch += 1
        self._apply()

    def set_epoch(self, epoch: int):
        self.cur_epoch = epoch
        self._apply()

    def _apply(self):
        ep = self.cur_epoch
        if ep <= self.warmup_ep:
            t = ep / self.warmup_ep
            lr = self.warm_lr0 + (self.base_lr - self.warm_lr0) * t
        else:
            t = (ep - self.warmup_ep) / max(1, self.total_ep - self.warmup_ep)
            lr = self.warm_lr0 + 0.5 * (self.base_lr - self.warm_lr0) * (
                1 + math.cos(math.pi * t))
        for pg in self.opt.param_groups:
            pg["lr"] = lr


# =============================================================================
# Train one epoch — returns (avg_loss, breakdown_dict)
# =============================================================================
def train_one_epoch(model: nn.Module, optimizer, loader, scaler,
                    device, epoch: int, ema: Optional[ModelEMA] = None,
                    grad_clip: float = 5.0):
    """Train for one epoch. Returns (avg_loss, breakdown_avg)."""
    model.train()
    total = 0.0
    n = 0
    breakdown = {"loss_classifier": 0.0, "loss_box_reg": 0.0,
                 "loss_box_refine": 0.0,
                 "loss_box_scale_distill": 0.0,
                 "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0,
                 "loss_rpn_micro_rescue": 0.0,
                 "loss_fpn_micro_feature": 0.0,
                 "loss_quality": 0.0,
                 "loss_metric": 0.0}  # placeholder for future metric loss
    pcgrad_totals = {
        "conflict": 0.0,
        "cosine": 0.0,
        "auxiliary_norm_ratio": 0.0,
    }
    pcgrad_batches = 0
    pcgrad_scope_totals = {}
    pcgrad_scope_batches = {}
    micro_rescue_totals = {
        "batches": 0,
        "valid_batches": 0,
        "micro_gt": 0,
        "selected_gt": 0,
    }
    micro_feature_totals = {
        "batches": 0,
        "valid_batches": 0,
        "micro_gt": 0,
        "selected_gt": 0,
    }
    bar = tqdm(loader, desc=f"Epoch {epoch}")
    for step, (imgs, targets) in enumerate(bar):
        imgs = [i.to(device, non_blocking=True) for i in imgs]
        targets = [{k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]
        try:
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss_dict = model(imgs, targets)
                loss = sum(v for v in loss_dict.values()
                           if isinstance(v, torch.Tensor) and torch.isfinite(v))
            if isinstance(loss, int) or not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            use_roi_pcgrad = (
                getattr(
                    getattr(model, "roi_heads", None),
                    "_cbl_scale_distill_pcgrad",
                    False,
                )
                and "loss_box_scale_distill" in loss_dict
                and torch.isfinite(loss_dict["loss_box_scale_distill"])
            )
            use_rpn_pcgrad = (
                getattr(
                    getattr(model, "rpn", None),
                    "_micro_rescue_pcgrad",
                    False,
                )
                and "loss_rpn_micro_rescue" in loss_dict
                and torch.isfinite(loss_dict["loss_rpn_micro_rescue"])
            )
            use_fpn_pcgrad = (
                getattr(
                    getattr(model, "backbone", None),
                    "_moc_feature_pcgrad",
                    False,
                )
                and "loss_fpn_micro_feature" in loss_dict
                and torch.isfinite(loss_dict["loss_fpn_micro_feature"])
            )
            if use_roi_pcgrad and (use_rpn_pcgrad or use_fpn_pcgrad):
                raise RuntimeError(
                    "RoI PCGrad cannot share a run with RPN or FPN PCGrad")
            if use_roi_pcgrad or use_rpn_pcgrad or use_fpn_pcgrad:
                auxiliary_specs = []
                if use_roi_pcgrad:
                    auxiliary_specs.append((
                        "roi",
                        loss_dict["loss_box_scale_distill"],
                        model.roi_heads.box_head.parameters(),
                    ))
                if use_rpn_pcgrad:
                    auxiliary_specs.append((
                        "rpn",
                        loss_dict["loss_rpn_micro_rescue"],
                        model.rpn.head.parameters(),
                    ))
                if use_fpn_pcgrad:
                    auxiliary_specs.append((
                        "fpn",
                        loss_dict["loss_fpn_micro_feature"],
                        model.backbone.fpn.parameters(),
                    ))
                auxiliary_keys = {
                    "roi": "loss_box_scale_distill",
                    "rpn": "loss_rpn_micro_rescue",
                    "fpn": "loss_fpn_micro_feature",
                }
                active_auxiliary_keys = {
                    auxiliary_keys[name] for name, _, _ in auxiliary_specs}
                detector_loss = sum(
                    value
                    for name, value in loss_dict.items()
                    if name not in active_auxiliary_keys
                    and isinstance(value, torch.Tensor)
                    and torch.isfinite(value)
                )
                pcgrad_results = _backward_with_disjoint_pcgrad(
                    detector_loss,
                    auxiliary_specs,
                    scaler,
                )
                for metrics in pcgrad_results.values():
                    for key, value in metrics.items():
                        pcgrad_totals[key] += value / len(pcgrad_results)
                pcgrad_batches += 1
                for scope, metrics in pcgrad_results.items():
                    totals = pcgrad_scope_totals.setdefault(scope, {
                        "conflict": 0.0,
                        "cosine": 0.0,
                        "auxiliary_norm_ratio": 0.0,
                    })
                    for key, value in metrics.items():
                        totals[key] += value
                    pcgrad_scope_batches[scope] = (
                        pcgrad_scope_batches.get(scope, 0) + 1)
                if use_rpn_pcgrad:
                    stats = getattr(model.rpn, "_micro_rescue_stats", {})
                    micro_rescue_totals["batches"] += 1
                    micro_rescue_totals["micro_gt"] += int(
                        stats.get("micro_gt", 0))
                    selected_gt = int(stats.get("selected_gt", 0))
                    micro_rescue_totals["selected_gt"] += selected_gt
                    micro_rescue_totals["valid_batches"] += int(
                        selected_gt > 0)
                if use_fpn_pcgrad:
                    stats = getattr(model.backbone, "_moc_feature_stats", {})
                    micro_feature_totals["batches"] += 1
                    micro_feature_totals["micro_gt"] += int(
                        stats.get("micro_gt", 0))
                    selected_gt = int(stats.get("selected_gt", 0))
                    micro_feature_totals["selected_gt"] += selected_gt
                    micro_feature_totals["valid_batches"] += int(
                        selected_gt > 0)
            else:
                scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)
            if (device.type == "cuda" and EMPTY_CACHE_EVERY > 0
                    and (step + 1) % EMPTY_CACHE_EVERY == 0):
                torch.cuda.empty_cache()
            total += loss.item()
            n += 1
            for k in breakdown:
                if k in loss_dict:
                    v = loss_dict[k]
                    if isinstance(v, torch.Tensor) and torch.isfinite(v):
                        breakdown[k] += float(v.detach())
            bar.set_postfix(loss=f"{loss.item():.4f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            raise
    avg = total / max(n, 1)
    breakdown_avg = {k: v / max(n, 1) for k, v in breakdown.items()}
    if pcgrad_batches:
        breakdown_avg.update({
            "pcgrad_conflict_rate": (
                pcgrad_totals["conflict"] / pcgrad_batches),
            "pcgrad_cosine": pcgrad_totals["cosine"] / pcgrad_batches,
            "pcgrad_auxiliary_norm_ratio": (
                pcgrad_totals["auxiliary_norm_ratio"] / pcgrad_batches),
            "pcgrad_batches": pcgrad_batches,
        })
        for scope, totals in pcgrad_scope_totals.items():
            batches = pcgrad_scope_batches[scope]
            breakdown_avg.update({
                f"pcgrad_{scope}_conflict_rate": (
                    totals["conflict"] / batches),
                f"pcgrad_{scope}_cosine": totals["cosine"] / batches,
                f"pcgrad_{scope}_auxiliary_norm_ratio": (
                    totals["auxiliary_norm_ratio"] / batches),
                f"pcgrad_{scope}_batches": batches,
            })
    if micro_rescue_totals["batches"]:
        micro_gt = micro_rescue_totals["micro_gt"]
        breakdown_avg.update({
            "micro_rescue_valid_batch_rate": (
                micro_rescue_totals["valid_batches"]
                / micro_rescue_totals["batches"]),
            "micro_rescue_selection_coverage": (
                micro_rescue_totals["selected_gt"] / micro_gt
                if micro_gt else 0.0),
            "micro_rescue_selected_gt": micro_rescue_totals["selected_gt"],
            "micro_rescue_micro_gt": micro_gt,
        })
    if micro_feature_totals["batches"]:
        micro_gt = micro_feature_totals["micro_gt"]
        breakdown_avg.update({
            "micro_feature_valid_batch_rate": (
                micro_feature_totals["valid_batches"]
                / micro_feature_totals["batches"]),
            "micro_feature_selection_coverage": (
                micro_feature_totals["selected_gt"] / micro_gt
                if micro_gt else 0.0),
            "micro_feature_selected_gt": micro_feature_totals["selected_gt"],
            "micro_feature_micro_gt": micro_gt,
        })
    return avg, breakdown_avg


# =============================================================================
# Build optimizer + scaler + scheduler
# =============================================================================
def build_optim_sched(model: nn.Module, lr: float, momentum: float,
                      weight_decay: float, warmup_start_lr: float,
                      warmup_epochs: int, total_epochs: int,
                      scheduler_type: str = "cosine",
                      lr_steps=None, lr_gamma=0.1):
    """Build optimizer, scaler, scheduler."""
    opt = torch.optim.SGD(model.parameters(), lr=warmup_start_lr,
                          momentum=momentum, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    if scheduler_type == "cosine":
        sched = WarmupCosineLR(opt, warmup_epochs=warmup_epochs,
                               total_epochs=total_epochs,
                               base_lr=lr, warmup_start_lr=warmup_start_lr)
        # Apply epoch 0 (just sets LR to warmup_start)
        sched.step_epoch()
    else:
        sched = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=lr_steps or [14, 18], gamma=lr_gamma)
        for pg in opt.param_groups:
            pg["lr"] = lr
    return opt, scaler, sched
