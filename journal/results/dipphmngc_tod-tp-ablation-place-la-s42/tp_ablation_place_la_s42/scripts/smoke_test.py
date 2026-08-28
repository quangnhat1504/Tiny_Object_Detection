"""Smoke test: 1 batch forward+backward trên GPU cho cả 3 loss types."""
import sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from common.config import DEVICE, seed_all, BATCH_SIZE, NUM_WORKERS
from common.dataset import build_training_datasets, collate_fn, compute_reliability_threshold
from common.metrics import get_metric_fn
from common.model import build_model
from torch.utils.data import DataLoader, WeightedRandomSampler

seed_all(42)

train_ds = build_training_datasets(use_patches=False, is_train=True)
metric_fn = get_metric_fn("sa_alw_full")
reliability_thr = compute_reliability_threshold(train_ds)
sampler = WeightedRandomSampler(train_ds.get_sample_weights(), len(train_ds), replacement=True)
loader = DataLoader(train_ds, batch_size=2, sampler=sampler, num_workers=0,
                    collate_fn=collate_fn, pin_memory=True, drop_last=True)

imgs, targets = next(iter(loader))
imgs = [i.to(DEVICE) for i in imgs]
targets = [{k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

print(f"Batch: {len(imgs)} images, first shape: {imgs[0].shape}")
print(f"Image device: {imgs[0].device}")

for loss_name in ["metric", "smooth_l1", "ciou", "diou"]:
    model = build_model(metric_fn, "la_loss", reliability_thr=reliability_thr,
                        box_loss_type=loss_name).to(DEVICE)
    model.train()
    model.roi_heads._current_epoch = 10  # post-warmup
    opt = torch.optim.SGD(model.parameters(), lr=0.001)
    opt.zero_grad()

    t0 = time.time()
    with torch.amp.autocast("cuda"):
        loss_dict = model(imgs, targets)
        loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
    loss.backward()
    opt.step()

    dt = time.time() - t0
    dev = next(model.parameters()).device
    print(f"  {loss_name:>10}: loss={loss.item():.4f}, device={dev}, time={dt:.1f}s, grad OK={next(model.parameters()).grad is not None}")
    del model; torch.cuda.empty_cache()

model = build_model(metric_fn, "la_loss", reliability_thr=reliability_thr,
                    box_loss_type="smooth_l1",
                    use_quality_score=True,
                    quality_loss_weight=0.5).to(DEVICE)
model.train()
model.roi_heads._current_epoch = 10
opt = torch.optim.SGD(model.parameters(), lr=0.001)
opt.zero_grad()
t0 = time.time()
with torch.amp.autocast("cuda"):
    loss_dict = model(imgs, targets)
    loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))
loss.backward()
opt.step()
dt = time.time() - t0
has_quality = "loss_quality" in loss_dict and torch.isfinite(loss_dict["loss_quality"])
q_loss_value = float(loss_dict["loss_quality"].detach()) if has_quality else 0.0
print(f"  {'smooth_l1+q':>10}: loss={loss.item():.4f}, q_loss={q_loss_value:.4f}, time={dt:.1f}s, quality OK={has_quality}")
del model; torch.cuda.empty_cache()

print("\nAll smoke tests PASSED on GPU")
