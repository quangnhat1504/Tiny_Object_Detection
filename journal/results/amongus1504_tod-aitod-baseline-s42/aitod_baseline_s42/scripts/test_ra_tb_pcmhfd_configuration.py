"""Contract tests for the audited RA-TB plus PC-MHFD configuration."""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train_frcnn_metric import _validate_ra_tb_pcmhfd_combination


def _validated_kwargs(teacher: Path) -> dict:
    return {
        "cbl_teacher": teacher,
        "micro_teacher": teacher,
        "coordinate_reliable": True,
        "head_only": False,
        "consensus_filter": False,
        "distill_distance": "teacher_bounded_gt",
        "cross_head": False,
        "cbl_pcgrad": False,
        "distill_stage": "refined",
        "feature_target": "high_frequency",
        "cbl_teacher_min_size": 960,
        "cbl_teacher_max_size": 1200,
        "micro_teacher_min_size": 960,
        "micro_teacher_max_size": 1200,
    }


def _expect_rejected(**kwargs) -> None:
    try:
        _validate_ra_tb_pcmhfd_combination(**kwargs)
    except ValueError:
        return
    raise AssertionError("Unaudited joint distillation configuration was accepted")


def main() -> None:
    with TemporaryDirectory() as temp_dir:
        teacher = Path(temp_dir) / "best.pt"
        kwargs = _validated_kwargs(teacher)
        assert _validate_ra_tb_pcmhfd_combination(**kwargs)

        _expect_rejected(**dict(
            kwargs, micro_teacher=Path(temp_dir) / "other.pt"))
        _expect_rejected(**dict(kwargs, feature_target="cosine"))
        _expect_rejected(**dict(kwargs, cbl_pcgrad=True))
        _expect_rejected(**dict(kwargs, micro_teacher_min_size=800))

        inactive = dict(kwargs, micro_teacher=None)
        assert not _validate_ra_tb_pcmhfd_combination(**inactive)
    print("RA-TB + PC-MHFD configuration contract: PASS")


if __name__ == "__main__":
    main()
