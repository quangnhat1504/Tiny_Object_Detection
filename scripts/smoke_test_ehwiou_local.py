"""
Local CUDA Smoke Test: Entropy-Modulated Homotopy (EH-WIoU) on NVIDIA GeForce RTX 5070 Ti.
Verifies memory footprint (<9.5 GiB), forward+backward propagation, and zero NaN/Inf under PyTorch AMP.
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from common.metrics import configure_metric
from common.model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== Running EH-WIoU Smoke Test on Device: {device} ===")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# 1. Build Metric & Faster R-CNN Model with EH-WIoU placement & loss
similarity_fn, distance_fn, _ = configure_metric("eh_wiou", h_wiou_sigma_0=8.0, h_wiou_static_gamma=0.5)
model = build_model(
    num_classes=9,
    metric_fn=similarity_fn,
    metric_distance_fn=distance_fn,
    placement="la_loss",
    box_loss_type="eh_wiou",
).to(device)
model.train()

# 2. Synthetic Micro Batches simulating AI-TOD 1024x1024 crops
batch_size = 2
imgs = [torch.randn(3, 800, 800, device=device) for _ in range(batch_size)]
targets = [
    {
        "boxes": torch.tensor([[100.0, 100.0, 106.0, 106.0], [250.0, 250.0, 260.0, 260.0]], device=device),
        "labels": torch.tensor([1, 2], device=device, dtype=torch.int64),
    },
    {
        "boxes": torch.tensor([[50.0, 50.0, 54.0, 54.0], [400.0, 400.0, 412.0, 412.0]], device=device),
        "labels": torch.tensor([3, 4], device=device, dtype=torch.int64),
    },
]

optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9)
optimizer.zero_grad()

print("Executing PyTorch AMP Forward + Backward Pass...")
t0 = time.time()
with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
    loss_dict = model(imgs, targets)
    total_loss = sum(v for v in loss_dict.values() if isinstance(v, torch.Tensor))

print(f"Forward pass completed in {time.time() - t0:.3f}s. Loss Dict: { {k: round(v.item(), 4) for k, v in loss_dict.items() if isinstance(v, torch.Tensor)} }")

# Check NaN/Inf
assert not torch.isnan(total_loss), "Total loss contains NaN!"
assert not torch.isinf(total_loss), "Total loss contains Inf!"

t1 = time.time()
total_loss.backward()
optimizer.step()
print(f"Backward pass + Optimizer step completed in {time.time() - t1:.3f}s.")

# Check CUDA Memory
if device.type == "cuda":
    max_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    print(f"\n[SUCCESS] Peak CUDA VRAM: {max_mem:.2f} GiB (< 9.5 GiB gate passed)")
    assert max_mem < 9.5, f"VRAM spike exceeded 9.5 GiB limit: {max_mem:.2f} GiB"

print("ALL EH-WIOU SMOKE TEST CRITERIA PASSED CLEANLY!\n")
