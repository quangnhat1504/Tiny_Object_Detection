---
title: CIoU/DIoU AMP Crash Fix for Tiny Boxes
type: topic
created: 2026-07-09
updated: 2026-07-09
sources: [model.py, train_utils.py]
tags: [ciou, amp, crash, float16, tiny-box, box-loss]
---

## Symptom

```
Epoch 8: 32%|███████████████████████████████████████████████████████▎  | 700/2214 [03:41<07:58,  3.16it/s, loss=0.8042]
Unhandled exception caught in c10/util/AbortHandler.h
00007FF941F0D956 torch_python.dll!torch::autograd::THPCppFunction_requires_grad
00007FF9BAF65D5B c10_cuda.dll!c10::cuda::c10_cuda_check_implementation
```

Hard crash (CUDA kernel segfault) at epoch 8-10 during `--box-loss ciou` training under AMP mixed precision.

## Root Cause

`torchvision.ops.complete_box_iou_loss()` receives float16 tensors under `torch.amp.autocast("cuda")`. For tiny boxes (2-8 px), float16:

1. Underflows area computations (w*h → 0)
2. Produces Inf/NaN in the CIoU aspect-ratio term (`v = 4/(pi^2) * (atan(w/gt_w) - atan(h/gt_h))^2`)
3. Backward pass CUDA kernel crashes on these degenerate values instead of producing NaN gradients

**Why casting inputs to .float() doesn't work:** The autocast context manager wraps the call, and `complete_box_iou_loss` internally creates intermediate tensors that inherit the autocast float16 context.

**Why `torch.isfinite()` guard doesn't help:** The crash happens inside the CUDA kernel before Python can see the output.

## Fix (model.py `_metric_box_loss`)

Three changes in the CIoU/DIoU branch:

1. **Disable autocast** for the entire CIoU block:
   ```python
   with torch.amp.autocast("cuda", enabled=False):
   ```

2. **Force float32** on all inputs to the box coder and CIoU loss:
   ```python
   box_reg_flat = box_regression_pos.float().reshape(K, num_classes * 4)
   decoded = box_coder.decode(box_reg_flat, [proposals_pos.float()])
   ```

3. **Filter degenerate boxes** before CIoU computation:
   ```python
   pred_area = pred_w * pred_h
   gt_area = gt_w * gt_h
   valid = (pred_area >= 4.0) & (gt_area >= 4.0)
   ```

4. **Fall back to zero loss** if no valid box pairs remain:
   ```python
   if valid.sum() < 1:
       iou_loss = torch.tensor(0.0, device=pred_boxes.device)
       metric_aux_val = torch.tensor(0.0, device=pred_boxes.device)
   ```

5. **Cast back** to autocast dtype for consistent `box_loss` construction.

## Affected Files

- `common/model.py` lines ~280-340 (`_metric_box_loss`, CIoU/DIoU branch)

## Alternative Solutions Considered

| Approach | Rejected Because |
|----------|-----------------|
| Run entire training in float32 | 2× slower, 2× more VRAM |
| Skip CIoU loss on float16 batches | Loss would be inconsistent epoch-to-epoch |
| PyTorch `torch.set_anomaly_detection()` | Only catches Python-level NaN, not CUDA segfaults |
| Raise `min=4.0` on all box clamps | Still fails when tiny boxes overlap with 0px gap |
