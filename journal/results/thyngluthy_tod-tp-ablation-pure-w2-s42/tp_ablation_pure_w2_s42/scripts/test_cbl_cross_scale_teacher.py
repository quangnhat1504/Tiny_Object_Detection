"""Real-data CUDA smoke for the high-resolution CBL teacher path."""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import DEVICE, seed_all
from common.dataset import build_training_datasets, collate_fn
from common.model import attach_cbl_cross_scale_teacher
from common.train_utils import _backward_with_pcgrad
from scripts.analyze_refinement_consistency import _build_model_from_checkpoint

DEFAULT_CHECKPOINT = (
    ROOT
    / ".runtime/kaggle/cbl_iterative_train_fair20/output/tod_output/runs"
    / "sa_alw_full__cbl__irtw0.5ir1s0.3__la_loss__seed42__cbl_iterative_train_fair20"
    / "best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--margin", type=float, default=0.02)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--head-only", action="store_true")
    parser.add_argument("--coordinate-reliable", action="store_true")
    parser.add_argument("--consensus-filter", action="store_true")
    parser.add_argument(
        "--distill-distance",
        choices=("kl", "ordered_w1", "teacher_bounded_gt"),
        default="kl",
    )
    parser.add_argument("--cross-head", action="store_true")
    parser.add_argument("--pcgrad", action="store_true")
    parser.add_argument(
        "--distill-stage", choices=("first", "refined"), default="first")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.ckpt.exists():
        raise FileNotFoundError(args.ckpt)
    seed_all(42)
    dataset = build_training_datasets(use_patches=False, is_train=True)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"),
    )
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    student, _ = _build_model_from_checkpoint(checkpoint, DEVICE)
    teacher, _ = _build_model_from_checkpoint(checkpoint, DEVICE)
    attach_cbl_cross_scale_teacher(
        student,
        teacher,
        loss_weight=0.25,
        temperature=2.0,
        advantage_margin=args.margin,
        teacher_min_size=960,
        teacher_max_size=1200,
        tiny_reference=16.0,
        tiny_weight_cap=2.0,
        head_only=args.head_only,
        coordinate_reliable=args.coordinate_reliable,
        consensus_filter=args.consensus_filter,
        distill_distance=args.distill_distance,
        cross_head=args.cross_head,
        pcgrad=args.pcgrad,
        distill_stage=args.distill_stage,
    )
    student.train()
    teacher.eval()
    assert not any(
        "_cbl_scale_teacher" in key for key in student.state_dict())
    optimizer = torch.optim.SGD(
        student.parameters(), lr=1e-5, momentum=0.9, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    if DEVICE.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    completed_steps = 0
    distillation_loss = None
    total_loss = None
    for images, targets in loader:
        images = [image.to(DEVICE) for image in images]
        targets = [
            {
                key: value.to(DEVICE) if isinstance(value, torch.Tensor) else value
                for key, value in target.items()
            }
            for target in targets
        ]
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
            losses = student(images, targets)
            total_loss = sum(losses.values())
        assert "loss_box_scale_distill" in losses
        distillation_loss = losses["loss_box_scale_distill"]
        assert torch.isfinite(distillation_loss) and distillation_loss > 0
        assert torch.isfinite(total_loss) and total_loss > 0
        isolated_gradients = torch.autograd.grad(
            distillation_loss,
            (
                student.roi_heads.box_predictor.bbox_dist.weight,
                student.roi_heads.box_head.fc7.weight,
                next(
                    parameter
                    for parameter in student.backbone.parameters()
                    if parameter.requires_grad
                ),
            ),
            retain_graph=True,
            allow_unused=True,
        )
        if args.cross_head:
            assert isolated_gradients[0] is None
        else:
            assert isolated_gradients[0] is not None
            assert isolated_gradients[0].abs().sum() > 0
        if args.head_only:
            assert isolated_gradients[1] is None
        else:
            assert isolated_gradients[1] is not None
            assert isolated_gradients[1].abs().sum() > 0
        if args.pcgrad:
            assert isolated_gradients[2] is None
            detector_loss = total_loss - distillation_loss
            pcgrad_metrics = _backward_with_pcgrad(
                detector_loss,
                distillation_loss,
                student.roi_heads.box_head.parameters(),
                scaler,
            )
            assert all(
                torch.isfinite(torch.tensor(value))
                for value in pcgrad_metrics.values()
            )
        else:
            scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        student_gradient = student.roi_heads.box_predictor.bbox_dist.weight.grad
        assert student_gradient is not None
        assert torch.isfinite(student_gradient).all()
        assert student_gradient.abs().sum() > 0
        assert all(parameter.grad is None for parameter in teacher.parameters())
        scaler.step(optimizer)
        scaler.update()
        assert student.roi_heads._cbl_scale_source_images is None
        completed_steps += 1
        if completed_steps >= args.steps:
            break
    assert completed_steps == args.steps

    student.eval()
    with tempfile.TemporaryDirectory() as temporary_directory:
        reload_path = Path(temporary_directory) / "cross_scale_reload.pt"
        torch.save(
            {
                "model": student.state_dict(),
                "config": checkpoint.get("config", {}),
            },
            reload_path,
        )
        reloaded_checkpoint = torch.load(
            reload_path, map_location="cpu", weights_only=False)
        reloaded, _ = _build_model_from_checkpoint(
            reloaded_checkpoint, DEVICE)
    with torch.no_grad():
        attached_output = student(images)
        reloaded_output = reloaded(images)
    for attached_item, reloaded_item in zip(
        attached_output, reloaded_output
    ):
        assert torch.equal(attached_item["labels"], reloaded_item["labels"])
        assert torch.allclose(
            attached_item["boxes"], reloaded_item["boxes"], atol=1e-5)
        assert torch.allclose(
            attached_item["scores"], reloaded_item["scores"], atol=1e-6)

    peak_gib = (
        torch.cuda.max_memory_allocated() / 1024**3
        if DEVICE.type == "cuda"
        else 0.0
    )
    print(
        "Cross-scale teacher integration PASSED: "
        f"loss={float(distillation_loss.detach()):.6f}, "
        f"total_loss={float(total_loss.detach()):.6f}, "
        f"peak_allocated={peak_gib:.3f} GiB, margin={args.margin:g}, "
        f"batch_size={args.batch_size}, steps={completed_steps}"
        f", head_only={args.head_only}"
        f", coordinate_reliable={args.coordinate_reliable}"
        f", consensus_filter={args.consensus_filter}"
        f", distill_distance={args.distill_distance}"
        f", cross_head={args.cross_head}"
        f", pcgrad={args.pcgrad}"
        f", distill_stage={args.distill_stage}"
    )


if __name__ == "__main__":
    main()
