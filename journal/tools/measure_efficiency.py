"""
Measure parameters, FLOPs, peak memory, and inference latency (FPS) for TOD methods.
"""
from __future__ import annotations
import json
import time
import sys
from pathlib import Path

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn

from common.model import build_model
from common.metrics import get_metric_fn, get_metric_distance_fn

ROOT = Path(r"C:\Users\ADMIN\_Project\tiny-object-detection")
OUT_JSON = ROOT / "journal/results/computational_efficiency_benchmark.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def measure_inference_fps(model: nn.Module, device: torch.device, input_size: tuple[int, int] = (800, 800), num_warmup: int = 10, num_runs: int = 50) -> tuple[float, float]:
    model.eval()
    dummy_input = [torch.randn(3, input_size[0], input_size[1], device=device)]
    
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy_input)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        for _ in range(num_runs):
            _ = model(dummy_input)
            
        if device.type == "cuda":
            torch.cuda.synchronize()
        total_time = time.perf_counter() - start
        
    fps = num_runs / total_time
    latency_ms = (total_time / num_runs) * 1000.0
    return fps, latency_ms


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    methods = [
        ("Faster R-CNN (Baseline)", None, "everywhere", "metric"),
        ("NWD (NeurIPS'21)", "nwd", "everywhere", "metric"),
        ("IGWD (TMM'22)", "igwd", "everywhere", "metric"),
        ("H-WIoU Proposed", "h_wiou", "h_wiou", "h_wiou"),
    ]
    
    results = {}
    
    print("\n" + "=" * 90)
    print(f"{'Method':<30} | {'Params (M)':<12} | {'Latency (ms)':<15} | {'FPS':<10} | {'Extra Params'}")
    print("-" * 90)
    
    base_params = None
    for name, metric_name, placement, box_loss in methods:
        metric_fn = get_metric_fn(metric_name) if metric_name else None
        metric_dist_fn = get_metric_distance_fn(metric_name) if metric_name else None
        
        model = build_model(
            num_classes=2,
            metric_fn=metric_fn,
            metric_distance_fn=metric_dist_fn,
            placement=placement,
            box_loss_type=box_loss,
        ).to(device)
        
        params = count_parameters(model)
        params_m = params / 1e6
        if base_params is None:
            base_params = params
            extra_params = "+0 (0.0%)"
        else:
            diff = params - base_params
            extra_params = f"+{diff}" if diff != 0 else "+0 (0.0%)"
            
        fps, latency_ms = measure_inference_fps(model, device)
        
        results[name] = {
            "parameters": params,
            "parameters_m": round(params_m, 2),
            "extra_parameters": extra_params,
            "inference_fps": round(fps, 2),
            "latency_ms": round(latency_ms, 2),
            "device": str(device),
        }
        
        print(f"{name:<30} | {params_m:<12.2f} | {latency_ms:<15.2f} | {fps:<10.2f} | {extra_params}")
        
    print("=" * 90)
    
    OUT_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark results to {OUT_JSON}")


if __name__ == "__main__":
    main()
