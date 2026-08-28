"""
Phase 2: Faster R-CNN with metric-based RPN + loss (la/la_loss/la_loss_nms).

Usage:
    python scripts/train_frcnn_metric.py --metric nwd --placement la_loss --seed 42
    python scripts/train_frcnn_metric.py --metric alw_full --placement la_loss --seed 123
    python scripts/train_frcnn_metric.py --metric sa_alw_full --placement la_loss --seed 2024
"""
from __future__ import annotations
import argparse
import csv
import sys
import time
import warnings
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from common.config import (
    EPOCHS, LR, MOMENTUM, WEIGHT_DECAY,
    WARMUP_EPOCHS, WARMUP_START_LR,
    BATCH_SIZE, NUM_WORKERS, DEVICE,
    USE_EMA, BOX_LOSS_WARMUP_EPOCHS,
    METRIC_BETA,
    CBL_ALPHA, CBL_NUM_BINS, CBL_GRID_BETA, CBL_UM_WEIGHT,
    MIN_SIZE, MAX_SIZE, RPN_BG_IOU,
    seed_all, make_output_dir,
)
from common.dataset import (
    collate_fn, build_training_datasets, build_tiled_datasets, build_copy_paste_pool,
    compute_reliability_threshold,
)
from common.metrics import CANONICAL_METRICS, NEEDS_RELIABILITY, configure_metric
from common.model import (
    attach_cbl_cross_scale_teacher,
    attach_pc_micro_object_feature_teacher,
    attach_pc_micro_rescue_rpn_teacher,
    build_model,
)
from common.train_utils import ModelEMA, WarmupCosineLR, train_one_epoch
from common.eval_utils import evaluate
from paper_a.evaluation.program_b_tiled import evaluate_tiled_model
from paper_a.evaluation.tinyperson_official import evaluate_tinyperson_official


def _select_evaluation_model(model, ema):
    """Return the exact model used for validation and its checkpoint label."""
    if ema is None:
        return model, "raw"
    return ema.get_model(), "ema"


def _set_transform_sizes(model, min_sizes, max_size):
    model.transform.min_size = tuple(int(size) for size in min_sizes)
    model.transform.max_size = int(max_size)


def _parse_snip_valid_range(value: str) -> tuple[float, float]:
    try:
        lower_text, upper_text = value.split(":", maxsplit=1)
        lower = float(lower_text)
        upper = (
            float("inf")
            if upper_text.lower() in {"inf", "infinity"}
            else float(upper_text)
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "SNIP ranges must use LOWER:UPPER, for example 20:inf"
        ) from error
    if lower < 0 or upper <= lower:
        raise argparse.ArgumentTypeError(
            "SNIP ranges require 0 <= LOWER < UPPER")
    return lower, upper


def _validate_ra_tb_pcmhfd_combination(
    *,
    cbl_teacher: Path | None,
    micro_teacher: Path | None,
    coordinate_reliable: bool,
    head_only: bool,
    consensus_filter: bool,
    distill_distance: str,
    cross_head: bool,
    cbl_pcgrad: bool,
    distill_stage: str,
    feature_target: str,
    cbl_teacher_min_size: int,
    cbl_teacher_max_size: int,
    micro_teacher_min_size: int,
    micro_teacher_max_size: int,
) -> bool:
    """Validate the only audited joint RoI/FPN distillation setup."""
    if cbl_teacher is None or micro_teacher is None:
        return False
    same_teacher = (
        Path(cbl_teacher).expanduser().resolve()
        == Path(micro_teacher).expanduser().resolve()
    )
    validated_configuration = bool(
        same_teacher
        and coordinate_reliable
        and not head_only
        and not consensus_filter
        and distill_distance == "teacher_bounded_gt"
        and not cross_head
        and not cbl_pcgrad
        and distill_stage == "refined"
        and feature_target == "high_frequency"
        and cbl_teacher_min_size == micro_teacher_min_size
        and cbl_teacher_max_size == micro_teacher_max_size
    )
    if not validated_configuration:
        raise ValueError(
            "Joint RoI/FPN distillation is allowed only for same-teacher "
            "RA-TB plus PC-MHFD with identical teacher transforms"
        )
    return True


def _validate_pc_mr_moc_combination(
    *,
    rpn_teacher: Path | None,
    fpn_teacher: Path | None,
    rpn_weight: float,
    fpn_weight: float,
    rpn_teacher_min_size: int,
    rpn_teacher_max_size: int,
    fpn_teacher_min_size: int,
    fpn_teacher_max_size: int,
    rpn_proposal_top_n: int,
    fpn_proposal_top_n: int,
    rpn_cutoff_px: float,
    fpn_cutoff_px: float,
    rpn_teacher_iou_floor: float,
    fpn_teacher_iou_floor: float,
    rpn_margin: float,
    fpn_margin: float,
    feature_target: str,
) -> bool:
    """Validate the frozen PC-MR-RPN plus PC-MOC-FD configuration."""
    if rpn_teacher is None or fpn_teacher is None:
        return False
    same_teacher = (
        Path(rpn_teacher).expanduser().resolve()
        == Path(fpn_teacher).expanduser().resolve()
    )
    validated_configuration = bool(
        same_teacher
        and rpn_weight == 0.005
        and fpn_weight == 0.15
        and rpn_teacher_min_size == fpn_teacher_min_size == 960
        and rpn_teacher_max_size == fpn_teacher_max_size == 1200
        and rpn_proposal_top_n == fpn_proposal_top_n == 300
        and rpn_cutoff_px == fpn_cutoff_px == 8.0
        and rpn_teacher_iou_floor == fpn_teacher_iou_floor == 0.50
        and rpn_margin == fpn_margin == 0.02
        and feature_target == "cosine"
    )
    if not validated_configuration:
        raise ValueError(
            "Joint RPN/FPN supervision is allowed only for the audited "
            "same-teacher PC-MR-RPN plus PC-MOC-FD configuration"
        )
    return True


def _validate_paper_a_protocol(
    *,
    metric: str,
    placement: str,
    box_loss: str,
    checkpoint_selector: str,
    disallowed_components: dict[str, bool],
) -> None:
    if metric not in CANONICAL_METRICS:
        return
    if placement not in {"la", "loss", "la_loss"}:
        raise ValueError(
            "Paper A canonical methods allow assignment, regression, or joint "
            "placement with fixed IoU-NMS"
        )
    if box_loss != "metric":
        raise ValueError("Paper A canonical runs require the canonical metric loss")
    if checkpoint_selector != "coco_ap":
        raise ValueError(
            "Paper A canonical runs must select checkpoints by validation COCO AP"
        )
    active = sorted(name for name, enabled in disallowed_components.items() if enabled)
    if active:
        raise ValueError(
            "Paper A canonical runs cannot include out-of-scope components: "
            + ", ".join(active)
        )


def train_metric(metric: str, placement: str, seed: int, resume: bool = False,
                 box_loss: str = "metric", tag: str = "",
                 metric_beta: float = METRIC_BETA,
                 h_wiou_sigma_0: float = 8.0,
                 h_wiou_form: str = "rational",
                 h_wiou_static_gamma: float = 0.5,
                 h_wiou_sigmoid_tau: float = 2.0,
                 sa_s_min: float | None = None,
                 sa_s_max: float | None = None,
                 sa_beta_min: float | None = None,
                 sa_beta_max: float | None = None,
                 sa_pos_weight_min: float | None = None,
                 sa_pos_weight_max: float | None = None,
                 sa_schedule_form: str = "linear",
                 checkpoint_selector: str = "map50",
                 box_loss_warmup_epochs: int | None = None,
                 quality_score: bool = False,
                 quality_loss_weight: float = 0.5,
                 quality_focal: bool = False,
                 quality_focal_beta: float = 2.0,
                 rank_sort: bool = False,
                 rank_sort_delta: float = 0.5,
                 double_head: bool = False,
                 double_head_reg_roi_scale: float = 1.3,
                 double_head_num_convs: int = 4,
                 cbl_refine_train_weight: float = 0.0,
                 cbl_refine_steps: int = 0,
                 cbl_refine_blend: float = 1.0,
                 cbl_refine_last_step_blend: float | None = None,
                 cbl_refine_last_center_blend: float | None = None,
                 cbl_refine_last_size_blend: float | None = None,
                 cbl_refine_score_threshold: float = 0.0,
                 cbl_refine_extra_min_size_ratio: float = 0.0,
                 rpn_refine_steps: int = 0,
                 rpn_refine_min_size_ratio: float = 0.0,
                 rpn_quality_objectness: bool = False,
                 rpn_quality_beta: float = 2.0,
                 rpn_quality_preserve_below_size_ratio: float = 0.0,
                 rpn_cascade: bool = False,
                 rpn_cascade_stage1_weight: float = 1.0,
                 rpn_iou_prediction: bool = False,
                 rpn_iou_prediction_loss_weight: float = 0.5,
                 rpn_iou_prediction_fusion_weight: float = 1.0,
                 rpn_iou_prediction_detached_tower: bool = False,
                 cbl_alpha: float = CBL_ALPHA,
                 cbl_num_bins: int = CBL_NUM_BINS,
                 cbl_grid_beta: float = CBL_GRID_BETA,
                 cbl_um_weight: float = CBL_UM_WEIGHT,
                 cbl_scale_distill_teacher: Path | None = None,
                 cbl_scale_distill_weight: float = 0.25,
                 cbl_scale_distill_temperature: float = 2.0,
                 cbl_scale_distill_margin: float = 0.02,
                 cbl_scale_distill_teacher_min_size: int = 960,
                 cbl_scale_distill_teacher_max_size: int = 1200,
                 cbl_scale_distill_tiny_reference: float = 16.0,
                 cbl_scale_distill_tiny_weight_cap: float = 2.0,
                 cbl_scale_distill_head_only: bool = False,
                 cbl_scale_distill_coordinate_reliable: bool = False,
                 cbl_scale_distill_consensus_filter: bool = False,
                 cbl_scale_distill_distance: str = "kl",
                 cbl_scale_distill_cross_head: bool = False,
                 cbl_scale_distill_pcgrad: bool = False,
                 cbl_scale_distill_stage: str = "first",
                 rpn_micro_rescue_teacher: Path | None = None,
                 rpn_micro_rescue_weight: float = 0.005,
                 rpn_micro_rescue_teacher_min_size: int = 960,
                 rpn_micro_rescue_teacher_max_size: int = 1200,
                 rpn_micro_rescue_proposal_top_n: int = 300,
                 rpn_micro_rescue_cutoff_px: float = 8.0,
                 rpn_micro_rescue_teacher_iou_floor: float = 0.50,
                 rpn_micro_rescue_margin: float = 0.02,
                 fpn_micro_feature_teacher: Path | None = None,
                 fpn_micro_feature_weight: float = 0.15,
                 fpn_micro_feature_teacher_min_size: int = 960,
                 fpn_micro_feature_teacher_max_size: int = 1200,
                 fpn_micro_feature_proposal_top_n: int = 300,
                 fpn_micro_feature_cutoff_px: float = 8.0,
                 fpn_micro_feature_teacher_iou_floor: float = 0.50,
                 fpn_micro_feature_margin: float = 0.02,
                 fpn_micro_feature_target: str = "cosine",
                 train_min_sizes: tuple[int, ...] | None = None,
                 train_max_size: int | None = None,
                 snip_valid_ranges: (
                     tuple[tuple[float, float], ...] | None
                 ) = None,
                 snip_rpn_ignore_iou_thresh: float = RPN_BG_IOU,
                 train_images: Path | None = None,
                 train_labels: Path | None = None,
                 validation_images: Path | None = None,
                 validation_labels: Path | None = None,
                 program_b_validation_manifest: Path | None = None,
                 program_b_validation_annotation: Path | None = None):
    explicit_data_paths = (train_images, train_labels, validation_images, validation_labels)
    if any(path is not None for path in explicit_data_paths) and any(path is None for path in explicit_data_paths):
        raise ValueError("Program B requires all four explicit tiled data paths")
    program_b_paths = (
        program_b_validation_manifest,
        program_b_validation_annotation,
    )
    if any(path is not None for path in program_b_paths) and any(path is None for path in program_b_paths):
        raise ValueError(
            "Program B original-image evaluation requires both manifest and annotation paths"
        )
    if any(path is not None for path in program_b_paths) and train_images is None:
        raise ValueError("Program B original-image evaluation requires explicit tiled data paths")
    train_min_sizes = tuple(train_min_sizes or (MIN_SIZE,))
    if any(size <= 0 for size in train_min_sizes):
        raise ValueError("Training minimum sizes must be positive")
    if train_max_size is None:
        scale_ratio = max(train_min_sizes) / MIN_SIZE
        train_max_size = max(MAX_SIZE, round(MAX_SIZE * scale_ratio))
    if train_max_size < max(train_min_sizes):
        raise ValueError(
            "Training max size must be at least the largest minimum size")
    if (
        snip_valid_ranges is not None
        and len(snip_valid_ranges) != len(train_min_sizes)
    ):
        raise ValueError(
            "SNIP valid ranges must match the number of training sizes")
    joint_ra_tb_pcmhfd = _validate_ra_tb_pcmhfd_combination(
        cbl_teacher=cbl_scale_distill_teacher,
        micro_teacher=fpn_micro_feature_teacher,
        coordinate_reliable=cbl_scale_distill_coordinate_reliable,
        head_only=cbl_scale_distill_head_only,
        consensus_filter=cbl_scale_distill_consensus_filter,
        distill_distance=cbl_scale_distill_distance,
        cross_head=cbl_scale_distill_cross_head,
        cbl_pcgrad=cbl_scale_distill_pcgrad,
        distill_stage=cbl_scale_distill_stage,
        feature_target=fpn_micro_feature_target,
        cbl_teacher_min_size=cbl_scale_distill_teacher_min_size,
        cbl_teacher_max_size=cbl_scale_distill_teacher_max_size,
        micro_teacher_min_size=fpn_micro_feature_teacher_min_size,
        micro_teacher_max_size=fpn_micro_feature_teacher_max_size,
    )
    joint_pc_mr_moc = _validate_pc_mr_moc_combination(
        rpn_teacher=rpn_micro_rescue_teacher,
        fpn_teacher=fpn_micro_feature_teacher,
        rpn_weight=rpn_micro_rescue_weight,
        fpn_weight=fpn_micro_feature_weight,
        rpn_teacher_min_size=rpn_micro_rescue_teacher_min_size,
        rpn_teacher_max_size=rpn_micro_rescue_teacher_max_size,
        fpn_teacher_min_size=fpn_micro_feature_teacher_min_size,
        fpn_teacher_max_size=fpn_micro_feature_teacher_max_size,
        rpn_proposal_top_n=rpn_micro_rescue_proposal_top_n,
        fpn_proposal_top_n=fpn_micro_feature_proposal_top_n,
        rpn_cutoff_px=rpn_micro_rescue_cutoff_px,
        fpn_cutoff_px=fpn_micro_feature_cutoff_px,
        rpn_teacher_iou_floor=rpn_micro_rescue_teacher_iou_floor,
        fpn_teacher_iou_floor=fpn_micro_feature_teacher_iou_floor,
        rpn_margin=rpn_micro_rescue_margin,
        fpn_margin=fpn_micro_feature_margin,
        feature_target=fpn_micro_feature_target,
    )
    if cbl_scale_distill_teacher is not None and box_loss != "cbl":
        raise ValueError("Cross-scale distillation requires CBL localization")
    if cbl_scale_distill_coordinate_reliable and cbl_scale_distill_teacher is None:
        raise ValueError(
            "Coordinate-reliable distillation requires a teacher checkpoint")
    if cbl_scale_distill_coordinate_reliable and cbl_scale_distill_head_only:
        raise ValueError(
            "Coordinate-reliable distillation requires shared-head adaptation")
    if (
        cbl_scale_distill_consensus_filter
        and not cbl_scale_distill_coordinate_reliable
    ):
        raise ValueError(
            "Consensus filtering requires coordinate-reliable distillation")
    if cbl_scale_distill_distance not in {
        "kl", "ordered_w1", "teacher_bounded_gt"
    }:
        raise ValueError(
            f"Unsupported cross-scale distance: {cbl_scale_distill_distance}")
    if cbl_scale_distill_cross_head and not cbl_scale_distill_coordinate_reliable:
        raise ValueError("Cross-head distillation requires coordinate reliability")
    if cbl_scale_distill_cross_head and (
        cbl_scale_distill_head_only
        or cbl_scale_distill_consensus_filter
        or cbl_scale_distill_distance != "kl"
    ):
        raise ValueError(
            "Cross-head Gate 0 must isolate the shared-representation path")
    if cbl_scale_distill_pcgrad and not cbl_scale_distill_cross_head:
        raise ValueError("PCGrad requires cross-head distillation")
    if cbl_scale_distill_stage not in {"first", "refined"}:
        raise ValueError(
            f"Unsupported cross-scale stage: {cbl_scale_distill_stage}")
    if cbl_scale_distill_stage == "refined":
        if cbl_scale_distill_teacher is None or cbl_refine_train_weight <= 0:
            raise ValueError(
                "Refined-stage distillation requires teacher and refinement")
        if (
            cbl_scale_distill_head_only
            or cbl_scale_distill_consensus_filter
            or cbl_scale_distill_cross_head
            or cbl_scale_distill_pcgrad
        ):
            raise ValueError(
                "Refined-stage Gate 0 must isolate coordinate-reliable KL")
    if rpn_micro_rescue_teacher is not None:
        if box_loss != "cbl":
            raise ValueError("PC-MR-RPN requires the iterative-CBL baseline")
        if cbl_scale_distill_teacher is not None:
            raise ValueError("PC-MR-RPN cannot share a run with RoI distillation")
        if (
            rpn_quality_objectness
            or rpn_cascade
            or rpn_iou_prediction
            or rpn_refine_steps > 0
        ):
            raise ValueError("PC-MR-RPN requires the standard RPN path")
        if rpn_micro_rescue_weight <= 0:
            raise ValueError("PC-MR-RPN weight must be positive")
        if (
            rpn_micro_rescue_teacher_min_size <= 0
            or rpn_micro_rescue_teacher_max_size
            < rpn_micro_rescue_teacher_min_size
        ):
            raise ValueError("Invalid PC-MR-RPN teacher transform")
        if (
            rpn_micro_rescue_proposal_top_n <= 0
            or rpn_micro_rescue_cutoff_px <= 0
            or not 0 <= rpn_micro_rescue_teacher_iou_floor <= 1
            or rpn_micro_rescue_margin < 0
        ):
            raise ValueError("Invalid PC-MR-RPN selection thresholds")
    if fpn_micro_feature_teacher is not None:
        micro_feature_method = (
            "PC-MHFD" if fpn_micro_feature_target == "high_frequency"
            else "PC-MOC-FD"
        )
        if box_loss != "cbl":
            raise ValueError(
                f"{micro_feature_method} requires the iterative-CBL baseline")
        if cbl_scale_distill_teacher is not None and not joint_ra_tb_pcmhfd:
            raise ValueError("PC-MOC-FD cannot share a run with RoI distillation")
        if rpn_micro_rescue_teacher is not None and not joint_pc_mr_moc:
            raise ValueError("PC-MOC-FD cannot share a run with PC-MR-RPN")
        if (
            rpn_quality_objectness
            or rpn_cascade
            or rpn_iou_prediction
            or rpn_refine_steps > 0
        ):
            raise ValueError("PC-MOC-FD requires the standard RPN path")
        if fpn_micro_feature_weight <= 0:
            raise ValueError(f"{micro_feature_method} weight must be positive")
        if fpn_micro_feature_target not in {"cosine", "high_frequency"}:
            raise ValueError(
                f"Unknown FPN micro feature target: {fpn_micro_feature_target}")
        if (
            fpn_micro_feature_teacher_min_size <= 0
            or fpn_micro_feature_teacher_max_size
            < fpn_micro_feature_teacher_min_size
        ):
            raise ValueError("Invalid PC-MOC-FD teacher transform")
        if (
            fpn_micro_feature_proposal_top_n <= 0
            or fpn_micro_feature_cutoff_px <= 0
            or not 0 <= fpn_micro_feature_teacher_iou_floor <= 1
            or fpn_micro_feature_margin < 0
        ):
            raise ValueError("Invalid PC-MOC-FD selection thresholds")

    _validate_paper_a_protocol(
        metric=metric,
        placement=placement,
        box_loss=box_loss,
        checkpoint_selector=checkpoint_selector,
        disallowed_components={
            "quality_score": quality_score,
            "quality_focal": quality_focal,
            "rank_sort": rank_sort,
            "double_head": double_head,
            "cbl_refinement": cbl_refine_train_weight > 0 or cbl_refine_steps > 0,
            "cbl_distillation": cbl_scale_distill_teacher is not None,
            "rpn_refinement": rpn_refine_steps > 0,
            "rpn_quality": rpn_quality_objectness,
            "rpn_cascade": rpn_cascade,
            "rpn_iou_prediction": rpn_iou_prediction,
            "pc_micro_rescue": rpn_micro_rescue_teacher is not None,
            "pc_micro_feature": fpn_micro_feature_teacher is not None,
        },
    )
    metric_fn, metric_distance_fn, metric_config = configure_metric(
        metric,
        beta=metric_beta,
        s_min=sa_s_min,
        s_max=sa_s_max,
        beta_min=sa_beta_min,
        beta_max=sa_beta_max,
        w_min=sa_pos_weight_min,
        w_max=sa_pos_weight_max,
        schedule_form=sa_schedule_form,
        h_wiou_sigma_0=h_wiou_sigma_0,
        h_wiou_form=h_wiou_form,
        h_wiou_static_gamma=h_wiou_static_gamma,
        h_wiou_sigmoid_tau=h_wiou_sigmoid_tau,
    )

    metric_name = metric if box_loss == "metric" else f"{metric}__{box_loss}"
    if metric == "h_wiou":
        if h_wiou_form != "rational":
            metric_name += f"__{h_wiou_form}"
        if h_wiou_sigma_0 != 8.0:
            metric_name += f"__sig{h_wiou_sigma_0:g}"
    if metric_config.get("schedule_source"):
        metric_name += (
            f"__s{metric_config['s_min']:g}-{metric_config['s_max']:g}"
            f"b{metric_config['beta_min']:g}-{metric_config['beta_max']:g}"
            f"w{metric_config['w_min']:g}-{metric_config['w_max']:g}"
            f"f{metric_config['schedule_form']}"
        )
    if quality_score:
        metric_name = f"{metric_name}__q{quality_loss_weight:g}"
    if quality_focal:
        metric_name = f"{metric_name}__qflb{quality_focal_beta:g}"
    if rank_sort:
        metric_name = f"{metric_name}__rsd{rank_sort_delta:g}"
    if double_head:
        metric_name = (
            f"{metric_name}__dh{double_head_num_convs}"
            f"s{double_head_reg_roi_scale:g}"
        )
    if cbl_refine_train_weight > 0:
        metric_name = (
            f"{metric_name}__irtw{cbl_refine_train_weight:g}"
            f"ir{cbl_refine_steps}s{cbl_refine_score_threshold:g}"
        )
        if cbl_refine_extra_min_size_ratio > 0:
            metric_name += f"m{cbl_refine_extra_min_size_ratio:g}"
    if cbl_scale_distill_teacher is not None:
        metric_name += (
            f"__csldw{cbl_scale_distill_weight:g}"
            f"t{cbl_scale_distill_temperature:g}"
            f"m{cbl_scale_distill_margin:g}"
        )
        if cbl_scale_distill_head_only:
            metric_name += "ho"
        if cbl_scale_distill_coordinate_reliable:
            metric_name += "cr"
        if cbl_scale_distill_consensus_filter:
            metric_name += "cf"
        if cbl_scale_distill_distance == "ordered_w1":
            metric_name += "ow1"
        elif cbl_scale_distill_distance == "teacher_bounded_gt":
            metric_name += "tbg"
        if cbl_scale_distill_cross_head:
            metric_name += "xh"
        if cbl_scale_distill_pcgrad:
            metric_name += "pc"
        if cbl_scale_distill_stage == "refined":
            metric_name += "r2"
    if rpn_micro_rescue_teacher is not None:
        metric_name += f"__pcmrrpnw{rpn_micro_rescue_weight:g}"
    if fpn_micro_feature_teacher is not None:
        micro_feature_tag = (
            "pcmhfd" if fpn_micro_feature_target == "high_frequency"
            else "pcmocfd"
        )
        metric_name += f"__{micro_feature_tag}w{fpn_micro_feature_weight:g}"
    if rpn_refine_steps > 0:
        metric_name = f"{metric_name}__rpnr{rpn_refine_steps}"
        if rpn_refine_min_size_ratio > 0:
            metric_name += f"m{rpn_refine_min_size_ratio:g}"
    if rpn_quality_objectness:
        metric_name = f"{metric_name}__rpnqflb{rpn_quality_beta:g}"
        if rpn_quality_preserve_below_size_ratio > 0:
            metric_name += (
                f"m{rpn_quality_preserve_below_size_ratio:g}")
    if rpn_cascade:
        metric_name = (
            f"{metric_name}__rpncasw{rpn_cascade_stage1_weight:g}")
    if rpn_iou_prediction:
        metric_name = (
            f"{metric_name}__rpniouw{rpn_iou_prediction_loss_weight:g}")
        if rpn_iou_prediction_fusion_weight != 1:
            metric_name += f"f{rpn_iou_prediction_fusion_weight:g}"
        if rpn_iou_prediction_detached_tower:
            metric_name += "dt"
    output_name = f"{metric_name}__{placement}__seed{seed}"
    if tag:
        output_name = f"{output_name}__{tag}"
    OUTPUT_DIR = ROOT / "runs" / output_name
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"METRIC ABLATION — {metric} @ {placement}")
    print(f"  Seed: {seed}")
    print(f"  Box loss: {box_loss}")
    print(f"  Checkpoint selector: {checkpoint_selector}")
    if metric_config.get("canonical"):
        print(f"  Canonical metric config: {metric_config}")
    if box_loss_warmup_epochs is not None:
        print(f"  Box loss warmup epochs: {box_loss_warmup_epochs}")
    print(f"  Quality score: {quality_score} (weight={quality_loss_weight:g})")
    print(f"  Quality focal: {quality_focal} (beta={quality_focal_beta:g})")
    print(f"  Rank & Sort: {rank_sort} (delta={rank_sort_delta:g})")
    print(
        f"  Double-Head: {double_head} "
        f"(scale={double_head_reg_roi_scale:g}, "
        f"bottlenecks={double_head_num_convs})"
    )
    print(
        f"  CBL iterative train: weight={cbl_refine_train_weight:g}; "
        f"inference steps={cbl_refine_steps}, "
        f"blend={cbl_refine_blend:g}, "
        f"last step blend={cbl_refine_last_step_blend}, "
        f"last center blend={cbl_refine_last_center_blend}, "
        f"last size blend={cbl_refine_last_size_blend}, "
        f"score threshold={cbl_refine_score_threshold:g}, "
        f"extra min size ratio={cbl_refine_extra_min_size_ratio:g}"
    )
    print(
        "  CBL cross-scale distillation: "
        f"teacher={cbl_scale_distill_teacher}; "
        f"weight={cbl_scale_distill_weight:g}, "
        f"temperature={cbl_scale_distill_temperature:g}, "
        f"margin={cbl_scale_distill_margin:g}, "
        "teacher transform="
        f"{cbl_scale_distill_teacher_min_size}/"
        f"{cbl_scale_distill_teacher_max_size}, "
        f"tiny reference/cap={cbl_scale_distill_tiny_reference:g}/"
        f"{cbl_scale_distill_tiny_weight_cap:g}, "
        f"head only={cbl_scale_distill_head_only}, "
        f"coordinate reliable={cbl_scale_distill_coordinate_reliable}, "
        f"consensus filter={cbl_scale_distill_consensus_filter}, "
        f"distance={cbl_scale_distill_distance}, "
        f"cross head={cbl_scale_distill_cross_head}, "
        f"PCGrad={cbl_scale_distill_pcgrad}, "
        f"stage={cbl_scale_distill_stage}"
    )
    print(
        "  PC-MR-RPN: "
        f"teacher={rpn_micro_rescue_teacher}; "
        f"weight={rpn_micro_rescue_weight:g}, "
        "teacher transform="
        f"{rpn_micro_rescue_teacher_min_size}/"
        f"{rpn_micro_rescue_teacher_max_size}, "
        f"top_n={rpn_micro_rescue_proposal_top_n}, "
        f"micro_cutoff={rpn_micro_rescue_cutoff_px:g}px, "
        f"teacher_iou_floor={rpn_micro_rescue_teacher_iou_floor:g}, "
        f"margin={rpn_micro_rescue_margin:g}"
    )
    print(
        "  FPN micro feature distillation: "
        f"teacher={fpn_micro_feature_teacher}; "
        f"target={fpn_micro_feature_target}, "
        f"weight={fpn_micro_feature_weight:g}, "
        "teacher transform="
        f"{fpn_micro_feature_teacher_min_size}/"
        f"{fpn_micro_feature_teacher_max_size}, "
        f"top_n={fpn_micro_feature_proposal_top_n}, "
        f"micro_cutoff={fpn_micro_feature_cutoff_px:g}px, "
        f"teacher_iou_floor={fpn_micro_feature_teacher_iou_floor:g}, "
        f"margin={fpn_micro_feature_margin:g}"
    )
    print(
        f"  RPN inference refinement steps: {rpn_refine_steps}; "
        f"min size ratio={rpn_refine_min_size_ratio:g}"
    )
    print(
        f"  RPN quality objectness: {rpn_quality_objectness}; "
        f"beta={rpn_quality_beta:g}; "
        "preserve below size ratio="
        f"{rpn_quality_preserve_below_size_ratio:g}"
    )
    print(
        f"  RPN cascade: {rpn_cascade}; "
        f"stage-1 weight={rpn_cascade_stage1_weight:g}"
    )
    print(
        f"  RPN IoU prediction: {rpn_iou_prediction}; "
        f"loss weight={rpn_iou_prediction_loss_weight:g}; "
        f"fusion weight={rpn_iou_prediction_fusion_weight:g}; "
        f"detached tower={rpn_iou_prediction_detached_tower}"
    )
    print(
        f"  Transform: train min={train_min_sizes}, "
        f"train max={train_max_size}; eval={MIN_SIZE}/{MAX_SIZE}"
    )
    if snip_valid_ranges is not None:
        print(
            "  SNIP ranges: "
            + ", ".join(
                f"{size}=[{lower:g},{upper:g}]"
                for size, (lower, upper) in zip(
                    train_min_sizes, snip_valid_ranges)
            )
            + f"; RPN invalid-IoU>={snip_rpn_ignore_iou_thresh:g}"
        )
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Resume: {resume}")
    print(f"{'='*70}\n")

    seed_all(seed)

    # ── Data ──
    if train_images is None:
        train_ds = build_training_datasets(use_patches=False, is_train=True)
        val_ds = build_training_datasets(use_patches=False, is_train=False)
    else:
        train_ds, val_ds = build_tiled_datasets(
            Path(train_images), Path(train_labels),
            Path(validation_images), Path(validation_labels),
        )
    cp_pool = build_copy_paste_pool(train_ds)
    if cp_pool:
        train_ds.copy_paste_pool = cp_pool

    # Reliability threshold (for ALW/SA-ALW)
    reliability_thr = 16.0
    if metric in NEEDS_RELIABILITY:
        reliability_thr = compute_reliability_threshold(train_ds)
        print(f"  reliability_thr = {reliability_thr:.2f}")

    sampler = WeightedRandomSampler(
        train_ds.get_sample_weights(), len(train_ds), replacement=True)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, sampler=sampler,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"), drop_last=True)
    val_loader = DataLoader(
        val_ds, batch_size=2, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn,
        pin_memory=(DEVICE.type == "cuda"))

    # ── Model ──
    model = build_model(
        metric_fn=metric_fn,
        metric_distance_fn=metric_distance_fn,
        placement=placement,
        reliability_thr=reliability_thr,
        box_loss_type=box_loss,
        box_loss_warmup_epochs=(
            BOX_LOSS_WARMUP_EPOCHS if box_loss_warmup_epochs is None
            else box_loss_warmup_epochs
        ),
        use_quality_score=quality_score,
        quality_loss_weight=quality_loss_weight,
        use_quality_focal=quality_focal,
        quality_focal_beta=quality_focal_beta,
        use_rank_sort=rank_sort,
        rank_sort_delta=rank_sort_delta,
        use_double_head=double_head,
        double_head_reg_roi_scale=double_head_reg_roi_scale,
        double_head_num_convs=double_head_num_convs,
        cbl_refine_train_weight=cbl_refine_train_weight,
        cbl_refine_steps=cbl_refine_steps,
        cbl_refine_blend=cbl_refine_blend,
        cbl_refine_last_step_blend=cbl_refine_last_step_blend,
        cbl_refine_last_center_blend=cbl_refine_last_center_blend,
        cbl_refine_last_size_blend=cbl_refine_last_size_blend,
        cbl_refine_score_threshold=cbl_refine_score_threshold,
        cbl_refine_extra_min_size_ratio=cbl_refine_extra_min_size_ratio,
        rpn_refine_steps=rpn_refine_steps,
        rpn_refine_min_size_ratio=rpn_refine_min_size_ratio,
        rpn_quality_objectness=rpn_quality_objectness,
        rpn_quality_beta=rpn_quality_beta,
        rpn_quality_preserve_below_size_ratio=(
            rpn_quality_preserve_below_size_ratio),
        rpn_cascade=rpn_cascade,
        rpn_cascade_stage1_weight=rpn_cascade_stage1_weight,
        rpn_iou_prediction=rpn_iou_prediction,
        rpn_iou_prediction_loss_weight=(
            rpn_iou_prediction_loss_weight),
        rpn_iou_prediction_fusion_weight=(
            rpn_iou_prediction_fusion_weight),
        rpn_iou_prediction_detached_tower=(
            rpn_iou_prediction_detached_tower),
        cbl_alpha=cbl_alpha,
        cbl_num_bins=cbl_num_bins,
        cbl_grid_beta=cbl_grid_beta,
        cbl_um_weight=cbl_um_weight,
        transform_min_sizes=train_min_sizes,
        transform_max_size=train_max_size,
        snip_valid_ranges=snip_valid_ranges,
        snip_rpn_ignore_iou_thresh=snip_rpn_ignore_iou_thresh,
    ).to(DEVICE)

    # ── Optimizer ──
    opt = torch.optim.SGD(
        model.parameters(), lr=WARMUP_START_LR,
        momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    sched = WarmupCosineLR(
        opt, warmup_epochs=WARMUP_EPOCHS, total_epochs=EPOCHS,
        base_lr=LR, warmup_start_lr=WARMUP_START_LR)
    sched.step_epoch()

    ema = ModelEMA(model) if USE_EMA else None

    # ── Resume checkpoint ──
    best_mAP50 = 0.0
    best_ap75 = 0.0
    best_coco_ap = 0.0
    best_epoch = 0
    best_ap75_epoch = 0
    best_coco_ap_epoch = 0
    start_epoch = 1
    history = []

    ckpt_path = OUTPUT_DIR / "last.pt"
    if resume and ckpt_path.exists():
        print(f"[RESUME] Loading checkpoint: {ckpt_path}")
        ck = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        if ema and ck.get("ema"):
            ema.load_state_dict(ck["ema"])
        sched.set_epoch(ck.get("epoch", 0) + 1)
        best_mAP50 = ck.get("best_mAP50", 0.0)
        best_ap75 = ck.get("best_coco_AP75", ck.get("best_ap75", 0.0))
        best_coco_ap = ck.get("best_coco_AP", 0.0)
        best_epoch = ck.get("best_epoch", 0)
        best_ap75_epoch = ck.get("best_ap75_epoch", 0)
        best_coco_ap_epoch = ck.get("best_coco_ap_epoch", 0)
        history = ck.get("history", [])
        start_epoch = ck.get("epoch", 0) + 1
        print(f"[RESUME] Resuming at epoch {start_epoch}, "
              f"best mAP@50={best_mAP50:.4f} @ ep{best_epoch}, "
              f"best AP75={best_ap75:.4f} @ ep{best_ap75_epoch}")
    elif resume:
        print(f"[RESUME] No checkpoint found at {ckpt_path}, starting fresh.")

    teacher_checkpoint_metadata = None
    shared_cbl_teacher = None
    shared_micro_teacher = None
    if cbl_scale_distill_teacher is not None:
        teacher_path = Path(cbl_scale_distill_teacher)
        if not teacher_path.exists():
            raise FileNotFoundError(
                f"Cross-scale teacher checkpoint not found: {teacher_path}")
        print(f"[CSLD] Loading frozen teacher: {teacher_path}")
        teacher_checkpoint = torch.load(
            teacher_path, map_location="cpu", weights_only=False)
        teacher = deepcopy(model)
        teacher.load_state_dict(teacher_checkpoint["model"])
        teacher.to(DEVICE)
        attach_cbl_cross_scale_teacher(
            model,
            teacher,
            loss_weight=cbl_scale_distill_weight,
            temperature=cbl_scale_distill_temperature,
            advantage_margin=cbl_scale_distill_margin,
            teacher_min_size=cbl_scale_distill_teacher_min_size,
            teacher_max_size=cbl_scale_distill_teacher_max_size,
            tiny_reference=cbl_scale_distill_tiny_reference,
            tiny_weight_cap=cbl_scale_distill_tiny_weight_cap,
            head_only=cbl_scale_distill_head_only,
            coordinate_reliable=cbl_scale_distill_coordinate_reliable,
            consensus_filter=cbl_scale_distill_consensus_filter,
            distill_distance=cbl_scale_distill_distance,
            cross_head=cbl_scale_distill_cross_head,
            pcgrad=cbl_scale_distill_pcgrad,
            distill_stage=cbl_scale_distill_stage,
        )
        shared_cbl_teacher = teacher
        teacher_checkpoint_metadata = {
            "path": str(teacher_path),
            "epoch": teacher_checkpoint.get("epoch"),
            "model_source": teacher_checkpoint.get("model_source"),
            "config": teacher_checkpoint.get("config", {}),
        }
        del teacher_checkpoint

    rpn_micro_rescue_teacher_metadata = None
    if rpn_micro_rescue_teacher is not None:
        teacher_path = Path(rpn_micro_rescue_teacher)
        if not teacher_path.exists():
            raise FileNotFoundError(
                f"PC-MR-RPN teacher checkpoint not found: {teacher_path}")
        print(f"[PC-MR-RPN] Loading frozen teacher: {teacher_path}")
        teacher_checkpoint = torch.load(
            teacher_path, map_location="cpu", weights_only=False)
        teacher = deepcopy(model)
        teacher.load_state_dict(teacher_checkpoint["model"])
        teacher.to(DEVICE)
        attach_pc_micro_rescue_rpn_teacher(
            model,
            teacher,
            loss_weight=rpn_micro_rescue_weight,
            teacher_min_size=rpn_micro_rescue_teacher_min_size,
            teacher_max_size=rpn_micro_rescue_teacher_max_size,
            proposal_top_n=rpn_micro_rescue_proposal_top_n,
            micro_cutoff_px=rpn_micro_rescue_cutoff_px,
            teacher_iou_floor=rpn_micro_rescue_teacher_iou_floor,
            advantage_margin=rpn_micro_rescue_margin,
        )
        rpn_micro_rescue_teacher_metadata = {
            "path": str(teacher_path),
            "epoch": teacher_checkpoint.get("epoch"),
            "model_source": teacher_checkpoint.get("model_source"),
            "config": teacher_checkpoint.get("config", {}),
        }
        shared_micro_teacher = teacher
        del teacher_checkpoint

    fpn_micro_feature_teacher_metadata = None
    if fpn_micro_feature_teacher is not None:
        teacher_path = Path(fpn_micro_feature_teacher)
        if not teacher_path.exists():
            raise FileNotFoundError(
                f"PC-MOC-FD teacher checkpoint not found: {teacher_path}")
        micro_feature_method = (
            "PC-MHFD" if fpn_micro_feature_target == "high_frequency"
            else "PC-MOC-FD"
        )
        if joint_ra_tb_pcmhfd:
            if shared_cbl_teacher is None or teacher_checkpoint_metadata is None:
                raise RuntimeError("RA-TB teacher was not initialized")
            print(
                f"[{micro_feature_method}] Reusing the frozen RA-TB teacher: "
                f"{teacher_path}"
            )
            teacher = shared_cbl_teacher
            fpn_micro_feature_teacher_metadata = dict(
                teacher_checkpoint_metadata)
            fpn_micro_feature_teacher_metadata["path"] = str(teacher_path)
        elif joint_pc_mr_moc:
            if (
                shared_micro_teacher is None
                or rpn_micro_rescue_teacher_metadata is None
            ):
                raise RuntimeError("PC-MR-RPN teacher was not initialized")
            print(
                f"[{micro_feature_method}] Reusing the frozen PC-MR-RPN "
                f"teacher: {teacher_path}"
            )
            teacher = shared_micro_teacher
            fpn_micro_feature_teacher_metadata = dict(
                rpn_micro_rescue_teacher_metadata)
            fpn_micro_feature_teacher_metadata["path"] = str(teacher_path)
        else:
            print(f"[{micro_feature_method}] Loading frozen teacher: {teacher_path}")
            teacher_checkpoint = torch.load(
                teacher_path, map_location="cpu", weights_only=False)
            teacher = deepcopy(model)
            teacher.load_state_dict(teacher_checkpoint["model"])
            teacher.to(DEVICE)
            fpn_micro_feature_teacher_metadata = {
                "path": str(teacher_path),
                "epoch": teacher_checkpoint.get("epoch"),
                "model_source": teacher_checkpoint.get("model_source"),
                "config": teacher_checkpoint.get("config", {}),
            }
            del teacher_checkpoint
        attach_pc_micro_object_feature_teacher(
            model,
            teacher,
            loss_weight=fpn_micro_feature_weight,
            teacher_min_size=fpn_micro_feature_teacher_min_size,
            teacher_max_size=fpn_micro_feature_teacher_max_size,
            proposal_top_n=fpn_micro_feature_proposal_top_n,
            micro_cutoff_px=fpn_micro_feature_cutoff_px,
            teacher_iou_floor=fpn_micro_feature_teacher_iou_floor,
            advantage_margin=fpn_micro_feature_margin,
            feature_target=fpn_micro_feature_target,
        )

    csv_path = OUTPUT_DIR / "metrics.csv"
    fields = ["epoch", "train_loss", "val_loss", "mAP_50", "mAP_primary",
              "coco_AP", "coco_AP50", "coco_AP75", "coco_AR100",
              "AP_micro", "AP_tiny", "AP_small", "AP_large",
              "pcgrad_conflict_rate", "pcgrad_cosine",
              "pcgrad_auxiliary_norm_ratio",
              "pcgrad_rpn_conflict_rate", "pcgrad_rpn_cosine",
              "pcgrad_rpn_auxiliary_norm_ratio",
              "pcgrad_fpn_conflict_rate", "pcgrad_fpn_cosine",
              "pcgrad_fpn_auxiliary_norm_ratio",
              "micro_rescue_valid_batch_rate",
              "micro_rescue_selection_coverage",
              "micro_rescue_selected_gt", "micro_rescue_micro_gt",
              "micro_feature_valid_batch_rate",
              "micro_feature_selection_coverage",
              "micro_feature_selected_gt", "micro_feature_micro_gt",
              "lr", "seconds"]
    effective_box_loss_warmup_epochs = (
        BOX_LOSS_WARMUP_EPOCHS if box_loss_warmup_epochs is None
        else box_loss_warmup_epochs
    )
    run_config = {
        "metric": metric,
        "metric_config": metric_config,
        "placement": placement,
        "seed": seed,
        "checkpoint_selector": checkpoint_selector,
        "box_loss": box_loss,
        "box_loss_warmup_epochs": effective_box_loss_warmup_epochs,
        "tag": tag,
        "reliability_thr": reliability_thr,
        "quality_score": quality_score,
        "quality_loss_weight": quality_loss_weight,
        "quality_focal": quality_focal,
        "quality_focal_beta": quality_focal_beta,
        "rank_sort": rank_sort,
        "rank_sort_delta": rank_sort_delta,
        "double_head": double_head,
        "double_head_reg_roi_scale": double_head_reg_roi_scale,
        "double_head_num_convs": double_head_num_convs,
        "cbl_refine_train_weight": cbl_refine_train_weight,
        "cbl_refine_steps": cbl_refine_steps,
        "cbl_refine_blend": cbl_refine_blend,
        "cbl_refine_last_step_blend": cbl_refine_last_step_blend,
        "cbl_refine_last_center_blend": cbl_refine_last_center_blend,
        "cbl_refine_last_size_blend": cbl_refine_last_size_blend,
        "cbl_refine_score_threshold": cbl_refine_score_threshold,
        "cbl_refine_extra_min_size_ratio": (
            cbl_refine_extra_min_size_ratio
        ),
        "rpn_refine_steps": rpn_refine_steps,
        "rpn_refine_min_size_ratio": rpn_refine_min_size_ratio,
        "rpn_quality_objectness": rpn_quality_objectness,
        "rpn_quality_beta": rpn_quality_beta,
        "rpn_quality_preserve_below_size_ratio": (
            rpn_quality_preserve_below_size_ratio),
        "rpn_cascade": rpn_cascade,
        "rpn_cascade_stage1_weight": rpn_cascade_stage1_weight,
        "rpn_iou_prediction": rpn_iou_prediction,
        "rpn_iou_prediction_loss_weight": (
            rpn_iou_prediction_loss_weight),
        "rpn_iou_prediction_fusion_weight": (
            rpn_iou_prediction_fusion_weight),
        "rpn_iou_prediction_detached_tower": (
            rpn_iou_prediction_detached_tower),
        "cbl_alpha": cbl_alpha,
        "cbl_num_bins": cbl_num_bins,
        "cbl_grid_beta": cbl_grid_beta,
        "cbl_um_weight": cbl_um_weight,
        "cbl_scale_distill_teacher": (
            str(cbl_scale_distill_teacher)
            if cbl_scale_distill_teacher is not None
            else None
        ),
        "cbl_scale_distill_weight": cbl_scale_distill_weight,
        "cbl_scale_distill_temperature": cbl_scale_distill_temperature,
        "cbl_scale_distill_margin": cbl_scale_distill_margin,
        "cbl_scale_distill_teacher_min_size": (
            cbl_scale_distill_teacher_min_size),
        "cbl_scale_distill_teacher_max_size": (
            cbl_scale_distill_teacher_max_size),
        "cbl_scale_distill_tiny_reference": (
            cbl_scale_distill_tiny_reference),
        "cbl_scale_distill_tiny_weight_cap": (
            cbl_scale_distill_tiny_weight_cap),
        "cbl_scale_distill_head_only": cbl_scale_distill_head_only,
        "cbl_scale_distill_coordinate_reliable": (
            cbl_scale_distill_coordinate_reliable),
        "cbl_scale_distill_consensus_filter": (
            cbl_scale_distill_consensus_filter),
        "cbl_scale_distill_distance": cbl_scale_distill_distance,
        "cbl_scale_distill_cross_head": cbl_scale_distill_cross_head,
        "cbl_scale_distill_pcgrad": cbl_scale_distill_pcgrad,
        "cbl_scale_distill_stage": cbl_scale_distill_stage,
        "cbl_scale_distill_teacher_metadata": teacher_checkpoint_metadata,
        "rpn_micro_rescue_teacher": (
            str(rpn_micro_rescue_teacher)
            if rpn_micro_rescue_teacher is not None else None),
        "rpn_micro_rescue_weight": rpn_micro_rescue_weight,
        "rpn_micro_rescue_teacher_min_size": (
            rpn_micro_rescue_teacher_min_size),
        "rpn_micro_rescue_teacher_max_size": (
            rpn_micro_rescue_teacher_max_size),
        "rpn_micro_rescue_proposal_top_n": (
            rpn_micro_rescue_proposal_top_n),
        "rpn_micro_rescue_cutoff_px": rpn_micro_rescue_cutoff_px,
        "rpn_micro_rescue_teacher_iou_floor": (
            rpn_micro_rescue_teacher_iou_floor),
        "rpn_micro_rescue_margin": rpn_micro_rescue_margin,
        "rpn_micro_rescue_teacher_metadata": (
            rpn_micro_rescue_teacher_metadata),
        "fpn_micro_feature_teacher": (
            str(fpn_micro_feature_teacher)
            if fpn_micro_feature_teacher is not None else None),
        "fpn_micro_feature_weight": fpn_micro_feature_weight,
        "fpn_micro_feature_teacher_min_size": (
            fpn_micro_feature_teacher_min_size),
        "fpn_micro_feature_teacher_max_size": (
            fpn_micro_feature_teacher_max_size),
        "fpn_micro_feature_proposal_top_n": (
            fpn_micro_feature_proposal_top_n),
        "fpn_micro_feature_cutoff_px": fpn_micro_feature_cutoff_px,
        "fpn_micro_feature_teacher_iou_floor": (
            fpn_micro_feature_teacher_iou_floor),
        "fpn_micro_feature_margin": fpn_micro_feature_margin,
        "fpn_micro_feature_target": fpn_micro_feature_target,
        "fpn_micro_feature_teacher_metadata": (
            fpn_micro_feature_teacher_metadata),
        "joint_pc_mr_moc": joint_pc_mr_moc,
        "train_min_sizes": list(train_min_sizes),
        "train_max_size": train_max_size,
        "eval_min_size": MIN_SIZE,
        "eval_max_size": MAX_SIZE,
        "snip_valid_ranges": (
            [list(valid_range) for valid_range in snip_valid_ranges]
            if snip_valid_ranges is not None
            else None
        ),
        "snip_rpn_ignore_iou_thresh": snip_rpn_ignore_iou_thresh,
        "use_ema": USE_EMA,
        "program_b_original_image_evaluation": {
            "manifest": str(program_b_validation_manifest)
            if program_b_validation_manifest else None,
            "annotation": str(program_b_validation_annotation)
            if program_b_validation_annotation else None,
            "score_threshold": 0.05,
            "nms_iou_threshold": 0.5,
            "max_detections": 200,
        },
    }

    for epoch in range(start_epoch, EPOCHS + 1):
        _set_transform_sizes(model, train_min_sizes, train_max_size)
        # Set current epoch on model for box loss warmup
        if hasattr(model, 'roi_heads'):
            model.roi_heads._current_epoch = epoch
        t0 = time.time()
        tloss, breakdown = train_one_epoch(
            model, opt, train_loader, scaler, DEVICE, epoch, ema=ema)
        sched.step_epoch()
        cur_lr = opt.param_groups[0]["lr"]

        eval_model, eval_model_source = _select_evaluation_model(model, ema)
        _set_transform_sizes(eval_model, (MIN_SIZE,), MAX_SIZE)
        met = evaluate(eval_model, val_loader, DEVICE, measure_fps_flag=(epoch == EPOCHS))
        if program_b_validation_manifest is not None:
            original_image = evaluate_tiled_model(
                eval_model,
                val_loader,
                DEVICE,
                program_b_validation_annotation,
                program_b_validation_manifest,
                val_ds.img_files,
                evaluator=evaluate_tinyperson_official,
                score_threshold=0.05,
                nms_iou_threshold=0.5,
                max_detections=200,
            )
            official_metrics = original_image["evaluation"]["metrics"]
            met["program_b_original_image"] = original_image
            met["mAP_50"] = float(official_metrics["AP50_all"])
            met["coco_AP75"] = float(official_metrics["AP75_all"])
            met["coco_AP50"] = float(official_metrics["AP50_all"])
        elapsed = time.time() - t0

        mAP50 = met.get("mAP_50", 0) or 0
        coco_ap = met.get("coco_AP", 0) or 0
        coco_ap75 = met.get("coco_AP75", 0) or 0
        print(f"  Epoch {epoch}/{EPOCHS} | {elapsed:.1f}s | "
              f"mAP@50={mAP50:.4f} | AP75={coco_ap75:.4f} | "
              f"best75={best_ap75:.4f} @ ep{best_ap75_epoch}")
        if "pcgrad_batches" in breakdown:
            print(
                "  PCGrad: "
                f"conflict={breakdown['pcgrad_conflict_rate']:.3f}, "
                f"cosine={breakdown['pcgrad_cosine']:.4f}, "
                "aux/det="
                f"{breakdown['pcgrad_auxiliary_norm_ratio']:.4f} "
                f"over {breakdown['pcgrad_batches']} batches"
            )
            for scope in ("rpn", "fpn"):
                key = f"pcgrad_{scope}_batches"
                if key in breakdown:
                    print(
                        f"    {scope.upper()}: "
                        f"conflict={breakdown[f'pcgrad_{scope}_conflict_rate']:.3f}, "
                        f"cosine={breakdown[f'pcgrad_{scope}_cosine']:.4f}, "
                        "aux/det="
                        f"{breakdown[f'pcgrad_{scope}_auxiliary_norm_ratio']:.4f}"
                    )
        if "micro_rescue_valid_batch_rate" in breakdown:
            print(
                "  PC-MR-RPN: "
                f"valid_batches={breakdown['micro_rescue_valid_batch_rate']:.3f}, "
                f"coverage={breakdown['micro_rescue_selection_coverage']:.3f}, "
                f"selected={breakdown['micro_rescue_selected_gt']}/"
                f"{breakdown['micro_rescue_micro_gt']}"
            )
        if "micro_feature_valid_batch_rate" in breakdown:
            print(
                "  PC-MOC-FD: "
                f"valid_batches={breakdown['micro_feature_valid_batch_rate']:.3f}, "
                f"coverage={breakdown['micro_feature_selection_coverage']:.3f}, "
                f"selected={breakdown['micro_feature_selected_gt']}/"
                f"{breakdown['micro_feature_micro_gt']}"
            )

        row = {
            "epoch": epoch, "train_loss": round(tloss, 6),
            "val_loss": met.get("val_loss", ""),
            "mAP_50": round(mAP50, 6),
            "mAP_primary": met.get("mAP_primary", ""),
            "coco_AP": met.get("coco_AP", ""),
            "coco_AP50": met.get("coco_AP50", ""),
            "coco_AP75": met.get("coco_AP75", ""),
            "coco_AR100": met.get("coco_AR100", ""),
            "AP_micro": met.get("AP_micro", ""),
            "AP_tiny": met.get("AP_tiny", ""),
            "AP_small": met.get("AP_small", ""),
            "AP_large": met.get("AP_large", ""),
            "pcgrad_conflict_rate": breakdown.get(
                "pcgrad_conflict_rate", ""),
            "pcgrad_cosine": breakdown.get("pcgrad_cosine", ""),
            "pcgrad_auxiliary_norm_ratio": breakdown.get(
                "pcgrad_auxiliary_norm_ratio", ""),
            "pcgrad_rpn_conflict_rate": breakdown.get(
                "pcgrad_rpn_conflict_rate", ""),
            "pcgrad_rpn_cosine": breakdown.get("pcgrad_rpn_cosine", ""),
            "pcgrad_rpn_auxiliary_norm_ratio": breakdown.get(
                "pcgrad_rpn_auxiliary_norm_ratio", ""),
            "pcgrad_fpn_conflict_rate": breakdown.get(
                "pcgrad_fpn_conflict_rate", ""),
            "pcgrad_fpn_cosine": breakdown.get("pcgrad_fpn_cosine", ""),
            "pcgrad_fpn_auxiliary_norm_ratio": breakdown.get(
                "pcgrad_fpn_auxiliary_norm_ratio", ""),
            "micro_rescue_valid_batch_rate": breakdown.get(
                "micro_rescue_valid_batch_rate", ""),
            "micro_rescue_selection_coverage": breakdown.get(
                "micro_rescue_selection_coverage", ""),
            "micro_rescue_selected_gt": breakdown.get(
                "micro_rescue_selected_gt", ""),
            "micro_rescue_micro_gt": breakdown.get(
                "micro_rescue_micro_gt", ""),
            "micro_feature_valid_batch_rate": breakdown.get(
                "micro_feature_valid_batch_rate", ""),
            "micro_feature_selection_coverage": breakdown.get(
                "micro_feature_selection_coverage", ""),
            "micro_feature_selected_gt": breakdown.get(
                "micro_feature_selected_gt", ""),
            "micro_feature_micro_gt": breakdown.get(
                "micro_feature_micro_gt", ""),
            "lr": cur_lr, "seconds": round(elapsed, 2),
        }
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
            w.writerow(row)

        if mAP50 > best_mAP50:
            best_mAP50 = mAP50
            best_epoch = epoch
        if coco_ap75 > best_ap75:
            best_ap75 = coco_ap75
            best_ap75_epoch = epoch
        if coco_ap > best_coco_ap:
            best_coco_ap = coco_ap
            best_coco_ap_epoch = epoch

        history.append(row)

        # always save last.pt for resume
        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "model_source": "raw",
            "eval_model_source": eval_model_source,
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict(),
            "ema": ema.state_dict() if ema else None,
            "metrics": met, "best_mAP50": best_mAP50,
            "best_coco_AP75": best_ap75,
            "best_coco_AP": best_coco_ap,
            "best_epoch": best_epoch,
            "best_ap75_epoch": best_ap75_epoch,
            "best_coco_ap_epoch": best_coco_ap_epoch,
            "history": history,
            "config": run_config,
        }, OUTPUT_DIR / "last.pt")

        if mAP50 == best_mAP50 and epoch == best_epoch:
            torch.save({
                "epoch": epoch, "model": eval_model.state_dict(),
                "model_source": eval_model_source,
                "optimizer": opt.state_dict(),
                "metrics": met, "best_mAP50": best_mAP50,
                "best_coco_AP75": best_ap75,
                "best_coco_AP": best_coco_ap,
                "config": run_config,
            }, OUTPUT_DIR / "best.pt")

        if coco_ap75 == best_ap75 and epoch == best_ap75_epoch:
            torch.save({
                "epoch": epoch, "model": eval_model.state_dict(),
                "model_source": eval_model_source,
                "optimizer": opt.state_dict(),
                "metrics": met, "best_mAP50": best_mAP50,
                "best_coco_AP75": best_ap75,
                "best_coco_AP": best_coco_ap,
                "config": run_config,
            }, OUTPUT_DIR / "best_ap75.pt")

        if coco_ap == best_coco_ap and epoch == best_coco_ap_epoch:
            torch.save({
                "epoch": epoch, "model": eval_model.state_dict(),
                "model_source": eval_model_source,
                "optimizer": opt.state_dict(),
                "metrics": met, "best_mAP50": best_mAP50,
                "best_coco_AP75": best_ap75,
                "best_coco_AP": best_coco_ap,
                "config": run_config,
            }, OUTPUT_DIR / "best_coco_ap.pt")

    print(f"\n{'='*70}")
    print(f"DONE: best mAP@50 = {best_mAP50:.4f} @ epoch {best_epoch}")
    print(f"      best AP75   = {best_ap75:.4f} @ epoch {best_ap75_epoch}")
    print(f"      best COCO AP= {best_coco_ap:.4f} @ epoch {best_coco_ap_epoch}")
    selected_score = (
        best_coco_ap if checkpoint_selector == "coco_ap" else best_mAP50
    )
    selected_epoch = (
        best_coco_ap_epoch if checkpoint_selector == "coco_ap" else best_epoch
    )
    selected_artifact = (
        "best_coco_ap.pt" if checkpoint_selector == "coco_ap" else "best.pt"
    )
    print(
        f"      selected ({checkpoint_selector}) = {selected_score:.4f} "
        f"@ epoch {selected_epoch} [{selected_artifact}]"
    )
    print(f"Logs: {csv_path}")
    print(f"{'='*70}\n")

    return selected_score


def main():
    parser = argparse.ArgumentParser(description="FRCNN metric ablation")
    parser.add_argument("--metric", type=str, required=True,
                        choices=["standard", "nwd", "igwd", "igwd_log_shape", "igwd_anisotropic_s",
                                  "alw_full", "sa_alw_beta_only", "sa_alw_full",
                                  "sa_alw_pos_only", "alw_canonical",
                                  "sa_alw_canonical",
                                  "sa_alw_canonical_beta_only",
                                  "sa_alw_canonical_pos_only",
                                  "h_wiou"],
                        help="Metric name")
    parser.add_argument("--placement", type=str, default="la_loss",
                        choices=["la", "loss", "la_loss", "la_loss_nms", "sda_decoupled", "h_wiou"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true",
                        help="Resume from last.pt checkpoint")
    parser.add_argument("--box-loss", type=str, default="metric",
                        choices=["metric", "smooth_l1", "side_smooth_l1", "ciou", "diou", "cbl", "h_wiou"],
                        help="Box regression loss type (decoupled from metric)")
    parser.add_argument("--box-loss-warmup-epochs", type=int, default=None,
                        help="Override metric-loss warmup epochs before decoupled box loss")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix for output dir")
    parser.add_argument("--metric-beta", type=float, default=METRIC_BETA,
                        help="Fixed ALW assignment-kernel beta")
    parser.add_argument("--h-wiou-sigma-0", type=float, default=8.0,
                        help="Characteristic microscopic scale constant for H-WIoU")
    parser.add_argument("--h-wiou-form", type=str, default="rational",
                        choices=["rational", "exponential", "sigmoid", "static", "pure_w2", "pure_iou"],
                        help="Homotopy deformation functional form")
    parser.add_argument("--h-wiou-static-gamma", type=float, default=0.5,
                        help="Constant gamma value when --h-wiou-form=static")
    parser.add_argument("--h-wiou-sigmoid-tau", type=float, default=2.0,
                        help="Temperature constant for sigmoid homotopy deformation")
    parser.add_argument("--sa-s-min", type=float, default=None,
                        help="Frozen train-derived lower target-scale bound")
    parser.add_argument("--sa-s-max", type=float, default=None,
                        help="Frozen train-derived upper target-scale bound")
    parser.add_argument("--sa-beta-min", type=float, default=None)
    parser.add_argument("--sa-beta-max", type=float, default=None)
    parser.add_argument("--sa-pos-weight-min", type=float, default=None)
    parser.add_argument("--sa-pos-weight-max", type=float, default=None)
    parser.add_argument(
        "--sa-schedule-form",
        choices=["linear", "log_linear"],
        default="linear",
        help="Clipped target-scale interpolation used by canonical SA-ALW",
    )
    parser.add_argument(
        "--checkpoint-selector",
        choices=["map50", "coco_ap"],
        default="map50",
        help="Frozen validation metric used to select the submission checkpoint",
    )
    parser.add_argument("--quality-score", action="store_true",
                        help="Enable auxiliary RoI localization-quality head")
    parser.add_argument("--quality-loss-weight", type=float, default=0.5,
                        help="Weight for quality IoU target loss")
    parser.add_argument("--quality-focal", action="store_true",
                        help="Train a joint class-IoU score with Quality Focal Loss")
    parser.add_argument("--quality-focal-beta", type=float, default=2.0,
                        help="Quality Focal Loss modulating exponent")
    parser.add_argument("--rank-sort", action="store_true",
                        help="Train sampled RoI classification with Rank & Sort loss")
    parser.add_argument("--rank-sort-delta", type=float, default=0.5,
                        help="Rank & Sort comparison smoothing width")
    parser.add_argument("--double-head", action="store_true",
                        help="Use a convolutional CBL box-regression head")
    parser.add_argument("--double-head-reg-roi-scale", type=float, default=1.3,
                        help="Proposal scale used for Double-Head regression")
    parser.add_argument("--double-head-num-convs", type=int, default=4,
                        help="Residual bottlenecks in Double-Head regression")
    parser.add_argument("--cbl-refine-train-weight", type=float, default=0.0,
                        help="Auxiliary shared-head second-pass CBL loss weight")
    parser.add_argument("--cbl-refine-steps", type=int, default=0,
                        help="Inference-time repeated CBL regression passes")
    parser.add_argument("--cbl-refine-blend", type=float, default=1.0,
                        help="Fraction of each inference refinement update")
    parser.add_argument(
        "--cbl-refine-last-step-blend",
        type=float,
        default=None,
        help="Final inference update fraction; defaults to the shared blend",
    )
    parser.add_argument(
        "--cbl-refine-last-center-blend",
        type=float,
        default=None,
        help="Final center-update fraction; defaults to final step blend",
    )
    parser.add_argument(
        "--cbl-refine-last-size-blend",
        type=float,
        default=None,
        help="Final width/height-update fraction; defaults to final step blend",
    )
    parser.add_argument("--cbl-refine-score-threshold", type=float, default=0.0,
                        help="Preserve detections below this score")
    parser.add_argument(
        "--cbl-refine-extra-min-size-ratio",
        type=float,
        default=0.0,
        help=(
            "After pass one, refine only boxes at or above this normalized "
            "sqrt-area size; zero disables the gate"
        ),
    )
    parser.add_argument("--cbl-alpha", type=float, default=CBL_ALPHA,
                        help="CBL normalized delta range")
    parser.add_argument("--cbl-num-bins", type=int, default=CBL_NUM_BINS,
                        help="CBL distribution bins per coordinate")
    parser.add_argument("--cbl-grid-beta", type=float, default=CBL_GRID_BETA,
                        help="CBL interval-nonuniform grid density")
    parser.add_argument("--cbl-um-weight", type=float, default=CBL_UM_WEIGHT,
                        help="CBL uncertainty matching loss weight")
    parser.add_argument(
        "--cbl-scale-distill-teacher",
        type=Path,
        default=None,
        help="Frozen CBL checkpoint used as a high-resolution teacher",
    )
    parser.add_argument(
        "--cbl-scale-distill-weight", type=float, default=0.25)
    parser.add_argument(
        "--cbl-scale-distill-temperature", type=float, default=2.0)
    parser.add_argument(
        "--cbl-scale-distill-margin", type=float, default=0.02)
    parser.add_argument(
        "--cbl-scale-distill-teacher-min-size", type=int, default=960)
    parser.add_argument(
        "--cbl-scale-distill-teacher-max-size", type=int, default=1200)
    parser.add_argument(
        "--cbl-scale-distill-tiny-reference", type=float, default=16.0)
    parser.add_argument(
        "--cbl-scale-distill-tiny-weight-cap", type=float, default=2.0)
    parser.add_argument(
        "--cbl-scale-distill-head-only",
        action="store_true",
        help="Stop distillation gradients at the shared RoI feature",
    )
    parser.add_argument(
        "--cbl-scale-distill-coordinate-reliable",
        action="store_true",
        help=(
            "Weight each distilled coordinate by detached teacher advantage "
            "and normalized teacher certainty"
        ),
    )
    parser.add_argument(
        "--cbl-scale-distill-consensus-filter",
        action="store_true",
        help=(
            "Downweight CR coordinates when original and horizontally "
            "flipped teacher distributions disagree"
        ),
    )
    parser.add_argument(
        "--cbl-scale-distill-distance",
        choices=("kl", "ordered_w1", "teacher_bounded_gt"),
        default="kl",
        help="Distribution distance for cross-scale CBL distillation",
    )
    parser.add_argument(
        "--cbl-scale-distill-cross-head",
        action="store_true",
        help=(
            "Distill through the frozen teacher distribution head into the "
            "student shared RoI representation"
        ),
    )
    parser.add_argument(
        "--cbl-scale-distill-pcgrad",
        action="store_true",
        help=(
            "Detach pooled RoI inputs from the backbone and project "
            "conflicting cross-head gradients on the student box head"
        ),
    )
    parser.add_argument(
        "--cbl-scale-distill-stage",
        choices=("first", "refined"),
        default="first",
        help="Apply cross-scale localization distillation at pass 1 or pass 2",
    )
    parser.add_argument(
        "--rpn-micro-rescue-teacher",
        type=Path,
        default=None,
        help="Frozen high-resolution teacher for PC-MR-RPN training",
    )
    parser.add_argument(
        "--rpn-micro-rescue-weight", type=float, default=0.005)
    parser.add_argument(
        "--rpn-micro-rescue-teacher-min-size", type=int, default=960)
    parser.add_argument(
        "--rpn-micro-rescue-teacher-max-size", type=int, default=1200)
    parser.add_argument(
        "--rpn-micro-rescue-proposal-top-n", type=int, default=300)
    parser.add_argument(
        "--rpn-micro-rescue-cutoff-px", type=float, default=8.0)
    parser.add_argument(
        "--rpn-micro-rescue-teacher-iou-floor", type=float, default=0.50)
    parser.add_argument(
        "--rpn-micro-rescue-margin", type=float, default=0.02)
    parser.add_argument(
        "--fpn-micro-feature-teacher",
        type=Path,
        default=None,
        help="Frozen high-resolution teacher for PC-MOC-FD training",
    )
    parser.add_argument(
        "--fpn-micro-feature-weight", type=float, default=0.15)
    parser.add_argument(
        "--fpn-micro-feature-teacher-min-size", type=int, default=960)
    parser.add_argument(
        "--fpn-micro-feature-teacher-max-size", type=int, default=1200)
    parser.add_argument(
        "--fpn-micro-feature-proposal-top-n", type=int, default=300)
    parser.add_argument(
        "--fpn-micro-feature-cutoff-px", type=float, default=8.0)
    parser.add_argument(
        "--fpn-micro-feature-teacher-iou-floor", type=float, default=0.50)
    parser.add_argument(
        "--fpn-micro-feature-margin", type=float, default=0.02)
    parser.add_argument(
        "--fpn-micro-feature-target",
        choices=("cosine", "high_frequency"),
        default="cosine",
    )
    parser.add_argument(
        "--train-min-sizes",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Stochastic shorter-side resize choices during training; "
            "validation remains fixed at the project default"
        ),
    )
    parser.add_argument(
        "--rpn-refine-steps",
        type=int,
        default=0,
        help="Extra inference-only applications of fixed RPN box deltas",
    )
    parser.add_argument(
        "--rpn-refine-min-size-ratio",
        type=float,
        default=0.0,
        help=(
            "Repeat RPN deltas only above this normalized proposal "
            "sqrt-area; zero disables the gate"
        ),
    )
    parser.add_argument(
        "--rpn-quality-objectness",
        action="store_true",
        help=(
            "Train RPN objectness with proposal-IoU targets and binary "
            "Quality Focal Loss"
        ),
    )
    parser.add_argument(
        "--rpn-quality-beta",
        type=float,
        default=2.0,
        help="Modulating exponent for RPN Quality Focal Loss",
    )
    parser.add_argument(
        "--rpn-quality-preserve-below-size-ratio",
        type=float,
        default=0.0,
        help=(
            "Keep binary-positive RPN targets for matched GT below this "
            "normalized sqrt-area; zero disables the scale guard"
        ),
    )
    parser.add_argument(
        "--rpn-cascade",
        action="store_true",
        help=(
            "Train a regression-only first RPN stage and rematch detached "
            "refined anchors for the standard RPN stage"
        ),
    )
    parser.add_argument(
        "--rpn-cascade-stage1-weight",
        type=float,
        default=1.0,
        help="Loss weight for cascade RPN stage-1 box regression",
    )
    parser.add_argument(
        "--rpn-iou-prediction",
        action="store_true",
        help=(
            "Add PAA-style positive IoU prediction and unified proposal "
            "ranking"
        ),
    )
    parser.add_argument(
        "--rpn-iou-prediction-loss-weight",
        type=float,
        default=0.5,
        help="Loss weight for positive-only RPN IoU prediction BCE",
    )
    parser.add_argument(
        "--rpn-iou-prediction-fusion-weight",
        type=float,
        default=1.0,
        help=(
            "Geometric blend strength from presence-only (0) to PAA "
            "presence-IoU ranking (1)"
        ),
    )
    parser.add_argument(
        "--rpn-iou-prediction-detached-tower",
        action="store_true",
        help=(
            "Use a separate IoU conv tower and stop its loss gradient at "
            "the backbone feature"
        ),
    )
    parser.add_argument(
        "--train-max-size",
        type=int,
        default=None,
        help=(
            "Training maximum image side; defaults to the project aspect "
            "ratio scaled to the largest training minimum size"
        ),
    )
    parser.add_argument(
        "--snip-valid-ranges",
        type=_parse_snip_valid_range,
        nargs="+",
        default=None,
        help=(
            "Scale-specific transformed sqrt-area ranges aligned with "
            "--train-min-sizes, for example 20:inf 12.5:50 0:30"
        ),
    )
    parser.add_argument(
        "--snip-rpn-ignore-iou-threshold",
        type=float,
        default=RPN_BG_IOU,
        help="Ignore negative RPN anchors overlapping invalid-scale GT",
    )
    parser.add_argument("--train-images", type=Path, default=None)
    parser.add_argument("--train-labels", type=Path, default=None)
    parser.add_argument("--validation-images", type=Path, default=None)
    parser.add_argument("--validation-labels", type=Path, default=None)
    parser.add_argument("--program-b-validation-manifest", type=Path, default=None)
    parser.add_argument("--program-b-validation-annotation", type=Path, default=None)
    args = parser.parse_args()

    train_metric(args.metric, args.placement, args.seed, args.resume,
                 box_loss=args.box_loss, tag=args.tag,
                 metric_beta=args.metric_beta,
                 h_wiou_sigma_0=args.h_wiou_sigma_0,
                 h_wiou_form=args.h_wiou_form,
                 h_wiou_static_gamma=args.h_wiou_static_gamma,
                 h_wiou_sigmoid_tau=args.h_wiou_sigmoid_tau,
                 sa_s_min=args.sa_s_min,
                 sa_s_max=args.sa_s_max,
                 sa_beta_min=args.sa_beta_min,
                 sa_beta_max=args.sa_beta_max,
                 sa_pos_weight_min=args.sa_pos_weight_min,
                 sa_pos_weight_max=args.sa_pos_weight_max,
                 sa_schedule_form=args.sa_schedule_form,
                 checkpoint_selector=args.checkpoint_selector,
                 box_loss_warmup_epochs=args.box_loss_warmup_epochs,
                 quality_score=args.quality_score,
                 quality_loss_weight=args.quality_loss_weight,
                 quality_focal=args.quality_focal,
                 quality_focal_beta=args.quality_focal_beta,
                 rank_sort=args.rank_sort,
                 rank_sort_delta=args.rank_sort_delta,
                 double_head=args.double_head,
                 double_head_reg_roi_scale=args.double_head_reg_roi_scale,
                 double_head_num_convs=args.double_head_num_convs,
                 cbl_refine_train_weight=args.cbl_refine_train_weight,
                 cbl_refine_steps=args.cbl_refine_steps,
                 cbl_refine_blend=args.cbl_refine_blend,
                 cbl_refine_last_step_blend=(
                     args.cbl_refine_last_step_blend),
                 cbl_refine_last_center_blend=(
                     args.cbl_refine_last_center_blend),
                 cbl_refine_last_size_blend=(
                     args.cbl_refine_last_size_blend),
                 cbl_refine_score_threshold=(
                     args.cbl_refine_score_threshold),
                 cbl_refine_extra_min_size_ratio=(
                     args.cbl_refine_extra_min_size_ratio),
                 rpn_refine_steps=args.rpn_refine_steps,
                 rpn_refine_min_size_ratio=(
                     args.rpn_refine_min_size_ratio),
                 rpn_quality_objectness=args.rpn_quality_objectness,
                 rpn_quality_beta=args.rpn_quality_beta,
                 rpn_quality_preserve_below_size_ratio=(
                     args.rpn_quality_preserve_below_size_ratio),
                 rpn_cascade=args.rpn_cascade,
                 rpn_cascade_stage1_weight=(
                     args.rpn_cascade_stage1_weight),
                 rpn_iou_prediction=args.rpn_iou_prediction,
                 rpn_iou_prediction_loss_weight=(
                     args.rpn_iou_prediction_loss_weight),
                 rpn_iou_prediction_fusion_weight=(
                     args.rpn_iou_prediction_fusion_weight),
                 rpn_iou_prediction_detached_tower=(
                     args.rpn_iou_prediction_detached_tower),
                 cbl_alpha=args.cbl_alpha,
                 cbl_num_bins=args.cbl_num_bins,
                 cbl_grid_beta=args.cbl_grid_beta,
                 cbl_um_weight=args.cbl_um_weight,
                 cbl_scale_distill_teacher=(
                     args.cbl_scale_distill_teacher),
                 cbl_scale_distill_weight=(
                     args.cbl_scale_distill_weight),
                 cbl_scale_distill_temperature=(
                     args.cbl_scale_distill_temperature),
                 cbl_scale_distill_margin=(
                     args.cbl_scale_distill_margin),
                 cbl_scale_distill_teacher_min_size=(
                     args.cbl_scale_distill_teacher_min_size),
                 cbl_scale_distill_teacher_max_size=(
                     args.cbl_scale_distill_teacher_max_size),
                 cbl_scale_distill_tiny_reference=(
                     args.cbl_scale_distill_tiny_reference),
                 cbl_scale_distill_tiny_weight_cap=(
                     args.cbl_scale_distill_tiny_weight_cap),
                 cbl_scale_distill_head_only=(
                     args.cbl_scale_distill_head_only),
                 cbl_scale_distill_coordinate_reliable=(
                     args.cbl_scale_distill_coordinate_reliable),
                 cbl_scale_distill_consensus_filter=(
                     args.cbl_scale_distill_consensus_filter),
                 cbl_scale_distill_distance=(
                     args.cbl_scale_distill_distance),
                 cbl_scale_distill_cross_head=(
                     args.cbl_scale_distill_cross_head),
                 cbl_scale_distill_pcgrad=(
                     args.cbl_scale_distill_pcgrad),
                  cbl_scale_distill_stage=(
                      args.cbl_scale_distill_stage),
                  rpn_micro_rescue_teacher=(
                      args.rpn_micro_rescue_teacher),
                  rpn_micro_rescue_weight=(
                      args.rpn_micro_rescue_weight),
                  rpn_micro_rescue_teacher_min_size=(
                      args.rpn_micro_rescue_teacher_min_size),
                  rpn_micro_rescue_teacher_max_size=(
                      args.rpn_micro_rescue_teacher_max_size),
                  rpn_micro_rescue_proposal_top_n=(
                      args.rpn_micro_rescue_proposal_top_n),
                  rpn_micro_rescue_cutoff_px=(
                      args.rpn_micro_rescue_cutoff_px),
                  rpn_micro_rescue_teacher_iou_floor=(
                      args.rpn_micro_rescue_teacher_iou_floor),
                  rpn_micro_rescue_margin=(
                      args.rpn_micro_rescue_margin),
                  fpn_micro_feature_teacher=(
                      args.fpn_micro_feature_teacher),
                  fpn_micro_feature_weight=(
                      args.fpn_micro_feature_weight),
                  fpn_micro_feature_teacher_min_size=(
                      args.fpn_micro_feature_teacher_min_size),
                  fpn_micro_feature_teacher_max_size=(
                      args.fpn_micro_feature_teacher_max_size),
                  fpn_micro_feature_proposal_top_n=(
                      args.fpn_micro_feature_proposal_top_n),
                  fpn_micro_feature_cutoff_px=(
                      args.fpn_micro_feature_cutoff_px),
                  fpn_micro_feature_teacher_iou_floor=(
                      args.fpn_micro_feature_teacher_iou_floor),
                  fpn_micro_feature_margin=(
                      args.fpn_micro_feature_margin),
                  fpn_micro_feature_target=(
                      args.fpn_micro_feature_target),
                  train_min_sizes=(
                     tuple(args.train_min_sizes)
                     if args.train_min_sizes is not None
                     else None
                 ),
                 train_max_size=args.train_max_size,
                 snip_valid_ranges=(
                     tuple(args.snip_valid_ranges)
                     if args.snip_valid_ranges is not None
                     else None
                 ),
                 snip_rpn_ignore_iou_thresh=(
                     args.snip_rpn_ignore_iou_threshold
                 ),
                 train_images=args.train_images,
                 train_labels=args.train_labels,
                 validation_images=args.validation_images,
                 validation_labels=args.validation_labels,
                 program_b_validation_manifest=args.program_b_validation_manifest,
                 program_b_validation_annotation=args.program_b_validation_annotation)


if __name__ == "__main__":
    main()
