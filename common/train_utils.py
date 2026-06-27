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
                 "loss_objectness": 0.0, "loss_rpn_box_reg": 0.0,
                 "loss_metric": 0.0}  # placeholder for future metric loss
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
            if not torch.isfinite(loss):
                optimizer.zero_grad()
                continue
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if ema:
                ema.update(model)
            if step % EMPTY_CACHE_EVERY == 0:
                torch.cuda.empty_cache()
            total += loss.item()
            n += 1
            for k in breakdown:
                if k in loss_dict and torch.isfinite(loss_dict[k]):
                    breakdown[k] += float(loss_dict[k])
            bar.set_postfix(loss=f"{loss.item():.4f}")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                continue
            raise
    avg = total / max(n, 1)
    breakdown_avg = {k: v / max(n, 1) for k, v in breakdown.items()}
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