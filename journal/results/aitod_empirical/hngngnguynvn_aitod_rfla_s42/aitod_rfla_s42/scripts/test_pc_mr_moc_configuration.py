"""Contract tests for the audited PC-MR-RPN plus PC-MOC-FD setup."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_frcnn_metric import _validate_pc_mr_moc_combination


def _validated_kwargs(teacher: Path) -> dict:
    return {
        "rpn_teacher": teacher,
        "fpn_teacher": teacher,
        "rpn_weight": 0.005,
        "fpn_weight": 0.15,
        "rpn_teacher_min_size": 960,
        "rpn_teacher_max_size": 1200,
        "fpn_teacher_min_size": 960,
        "fpn_teacher_max_size": 1200,
        "rpn_proposal_top_n": 300,
        "fpn_proposal_top_n": 300,
        "rpn_cutoff_px": 8.0,
        "fpn_cutoff_px": 8.0,
        "rpn_teacher_iou_floor": 0.50,
        "fpn_teacher_iou_floor": 0.50,
        "rpn_margin": 0.02,
        "fpn_margin": 0.02,
        "feature_target": "cosine",
    }


def _expect_rejected(**kwargs) -> None:
    try:
        _validate_pc_mr_moc_combination(**kwargs)
    except ValueError:
        return
    raise AssertionError("Unaudited PC-MR + PC-MOC configuration was accepted")


def main() -> None:
    with TemporaryDirectory() as temp_dir:
        teacher = Path(temp_dir) / "best.pt"
        kwargs = _validated_kwargs(teacher)
        assert _validate_pc_mr_moc_combination(**kwargs)

        _expect_rejected(**dict(
            kwargs, fpn_teacher=Path(temp_dir) / "other.pt"))
        _expect_rejected(**dict(kwargs, fpn_weight=0.20))
        _expect_rejected(**dict(kwargs, feature_target="high_frequency"))
        _expect_rejected(**dict(kwargs, fpn_teacher_min_size=800))
        _expect_rejected(**dict(kwargs, fpn_proposal_top_n=200))

        inactive = dict(kwargs, fpn_teacher=None)
        assert not _validate_pc_mr_moc_combination(**inactive)
    print("PC-MR-RPN + PC-MOC-FD configuration contract: PASS")


if __name__ == "__main__":
    main()
