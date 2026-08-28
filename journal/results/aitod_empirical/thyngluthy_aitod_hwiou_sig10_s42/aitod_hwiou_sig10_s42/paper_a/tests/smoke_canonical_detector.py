from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.metrics import configure_metric
from common.model import MetricRPN, build_model


DEFAULT_OUTPUT = ROOT / "paper_a" / "diagnostics" / "canonical_detector_smoke_seed42.json"
SYNTHETIC_SCHEDULE = {
    "s_min": 5.0,
    "s_max": 20.0,
    "beta_min": 8.0,
    "beta_max": 10.0,
    "w_min": 1.0,
    "w_max": 1.5,
}


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def build_canonical(placement: str, schedule_form: str = "linear") -> torch.nn.Module:
    similarity, distance, _ = configure_metric(
        "sa_alw_canonical",
        **SYNTHETIC_SCHEDULE,
        schedule_form=schedule_form,
    )
    return build_model(
        metric_fn=similarity,
        metric_distance_fn=distance,
        placement=placement,
        box_loss_type="metric",
    )


def synthetic_batch(device: torch.device):
    generator = torch.Generator(device="cpu").manual_seed(42)
    image = torch.rand((3, 256, 256), generator=generator).to(device)
    target = {
        "boxes": torch.tensor(
            [[48.0, 56.0, 58.0, 70.0], [132.0, 106.0, 151.0, 128.0]],
            device=device,
        ),
        "labels": torch.tensor([1, 2], dtype=torch.int64, device=device),
        "image_id": torch.tensor([0], dtype=torch.int64, device=device),
    }
    return [image], [target]


def run(output: Path, *, schedule_form: str = "linear") -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("Canonical detector smoke requires CUDA")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    device = torch.device("cuda")

    baseline = build_model(metric_fn=None, placement="everywhere")
    baseline_parameters = parameter_count(baseline)
    del baseline
    gc.collect()

    result = {
        "status": "PASS",
        "seed": 42,
        "device": torch.cuda.get_device_name(0),
        "schedule": {**SYNTHETIC_SCHEDULE, "schedule_form": schedule_form},
        "schedule_use": "synthetic_technical_smoke_only_not_frozen_for_training",
        "baseline_parameters": baseline_parameters,
        "placements": {},
    }
    joint_state = None

    for placement in ("la", "loss", "la_loss"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model = build_canonical(placement, schedule_form).to(device).train()
        parameters = parameter_count(model)
        if parameters != baseline_parameters:
            raise AssertionError(
                f"{placement} parameter count {parameters} != {baseline_parameters}"
            )
        is_metric_rpn = isinstance(model.rpn, MetricRPN)
        if is_metric_rpn != (placement in {"la", "la_loss"}):
            raise AssertionError(f"incorrect RPN placement for {placement}")
        has_metric_roi_loss = hasattr(model.roi_heads, "_box_loss_type")
        if has_metric_roi_loss != (placement in {"loss", "la_loss"}):
            raise AssertionError(f"incorrect RoI placement for {placement}")

        optimizer = torch.optim.SGD(model.parameters(), lr=1e-5, momentum=0.9)
        scaler = torch.amp.GradScaler("cuda")
        images, targets = synthetic_batch(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            losses = model(images, targets)
            total = sum(losses.values())
        if not torch.isfinite(total):
            raise AssertionError(f"non-finite total loss for {placement}")
        scaler.scale(total).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_values = {name: float(value.detach()) for name, value in losses.items()}
        if not all(torch.isfinite(torch.tensor(value)) for value in loss_values.values()):
            raise AssertionError(f"non-finite loss component for {placement}")
        result["placements"][placement] = {
            "parameter_count": parameters,
            "metric_rpn": is_metric_rpn,
            "metric_roi_loss": has_metric_roi_loss,
            "losses": loss_values,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
        }
        if placement == "la_loss":
            joint_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        del model, optimizer, scaler, images, targets, losses, total
        gc.collect()

    torch.cuda.empty_cache()
    reload_model = build_canonical("la_loss", schedule_form)
    incompatible = reload_model.load_state_dict(joint_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise AssertionError("strict canonical detector reload failed")
    result["strict_reload"] = {
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--schedule-form",
        choices=["linear", "log_linear"],
        default="linear",
    )
    args = parser.parse_args()
    result = run(args.output, schedule_form=args.schedule_form)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
