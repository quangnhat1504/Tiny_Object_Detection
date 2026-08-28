"""Build a self-contained Kaggle notebook for the CR-SC-CBL fair-20 run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = (
    ROOT
    / ".runtime/kaggle/cr_sc_cbl_fair20/kernel/cr_sc_cbl_fair20.ipynb"
)
TEACHER_SHA256 = "90043edfd278a51eef76c8494f4edae8e37127e78fc79dda9eee8071cc29769a"
MODULES = (
    ("common.config", "common/config.py"),
    ("common.metrics.iou", "common/metrics/iou.py"),
    ("common.metrics.nwd", "common/metrics/nwd.py"),
    ("common.metrics.igwd", "common/metrics/igwd.py"),
    ("common.metrics.alw", "common/metrics/alw.py"),
    ("common.metrics.sa_alw", "common/metrics/sa_alw.py"),
    ("common.metrics", "common/metrics/__init__.py"),
    ("common.dataset", "common/dataset.py"),
    ("common.assigner", "common/assigner.py"),
    ("common.model", "common/model.py"),
    ("common.train_utils", "common/train_utils.py"),
    ("common.eval_utils", "common/eval_utils.py"),
    ("scripts.train_frcnn_metric", "scripts/train_frcnn_metric.py"),
)


def _cell(source: str, *, cell_type: str = "code") -> dict:
    cell = {
        "id": hashlib.sha256(f"{cell_type}\0{source}".encode()).hexdigest()[:12],
        "cell_type": cell_type,
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if cell_type == "code":
        cell.update({"execution_count": None, "outputs": []})
    return cell


def _module_cell(module_name: str, relative_path: str) -> str:
    source = (ROOT / relative_path).read_text(encoding="utf-8")
    if module_name == "scripts.train_frcnn_metric":
        source = source.replace("sys.path.insert(0, str(ROOT))\n", "")
    if "'''" in source:
        raise ValueError(f"Embedded source contains reserved delimiter: {relative_path}")
    fake_path = f"/kaggle/working/cr_sc_cbl_project/{relative_path}"
    return (
        f"# Embedded source: {relative_path}\n"
        f"_exec_module(\n"
        f"    {module_name!r},\n"
        f"    r'''{source}''',\n"
        f"    {fake_path!r},\n"
        f")\n"
    )


def _source_manifest() -> tuple[list[dict], str]:
    manifest = []
    combined = hashlib.sha256()
    for module_name, relative_path in MODULES:
        payload = (ROOT / relative_path).read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        manifest.append({
            "module": module_name,
            "path": relative_path,
            "sha256": digest,
            "bytes": len(payload),
        })
        combined.update(relative_path.encode("utf-8"))
        combined.update(b"\0")
        combined.update(payload)
        combined.update(b"\0")
    return manifest, combined.hexdigest()


def _training_cell(
    *,
    method: str,
    seed: int,
    tag: str,
    epochs: int,
    manifest: list[dict],
    bundle_digest: str,
    teacher_sha256: str,
) -> str:
    is_candidate = method != "baseline"
    is_distillation = method in {
        "cr_sc_cbl", "ra_cr_sc_cbl", "ra_tb_cbl", "ra_tb_pcmhfd"}
    is_pc_micro_rescue = method in {"pc_mr_rpn", "pc_mr_moc"}
    is_pc_micro_feature = method in {
        "pc_moc_fd", "pc_mhfd", "ra_tb_pcmhfd", "pc_mr_moc"}
    micro_feature_target = (
        "high_frequency"
        if method in {"pc_mhfd", "ra_tb_pcmhfd"}
        else "cosine"
    )
    micro_feature_weight = (
        0.20 if method in {"pc_mhfd", "ra_tb_pcmhfd"} else 0.15)
    distill_stage = (
        "refined"
        if method in {"ra_cr_sc_cbl", "ra_tb_cbl", "ra_tb_pcmhfd"}
        else "first"
    )
    distill_distance = (
        "teacher_bounded_gt"
        if method in {"ra_tb_cbl", "ra_tb_pcmhfd"}
        else "kl"
    )
    method_labels = {
        "baseline": "Iterative-CBL baseline",
        "cr_sc_cbl": "CR-SC-CBL",
        "ra_cr_sc_cbl": "RA-CR-SC-CBL",
        "ra_tb_cbl": "RA-TB-CBL",
        "pc_mr_rpn": "PC-MR-RPN",
        "pc_moc_fd": "PC-MOC-FD",
        "pc_mhfd": "PC-MHFD",
        "ra_tb_pcmhfd": "RA-TB + PC-MHFD",
        "pc_mr_moc": "PC-MR-RPN + PC-MOC-FD",
    }
    method_label = method_labels[method]
    teacher_setup = (
        f"""expected_teacher_sha256 = {teacher_sha256!r}
teacher_matches = []
for candidate_path in Path('/kaggle/input').rglob('best.pt'):
    candidate_sha256 = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    if candidate_sha256 == expected_teacher_sha256:
        teacher_matches.append((candidate_path, candidate_sha256))
if len(teacher_matches) != 1:
    raise RuntimeError(f'Expected one exact fair20 teacher, got {{teacher_matches}}')
teacher_path, teacher_sha256 = teacher_matches[0]
teacher_checkpoint = torch.load(teacher_path, map_location='cpu', weights_only=False)
if teacher_checkpoint.get('epoch') != 5 or teacher_checkpoint.get('model_source') != 'ema':
    raise RuntimeError('Teacher must be exact fair20 EMA epoch-5 best.pt')
del teacher_checkpoint
print('teacher', teacher_path, 'sha256', teacher_sha256)
"""
        if is_candidate
        else "teacher_path = None\nteacher_sha256 = None\n"
    )
    distillation_args = (
        f"""        cbl_scale_distill_teacher=teacher_path,
        cbl_scale_distill_weight=0.25,
        cbl_scale_distill_temperature=2.0,
        cbl_scale_distill_margin=0.02,
        cbl_scale_distill_teacher_min_size=960,
        cbl_scale_distill_teacher_max_size=1200,
        cbl_scale_distill_tiny_reference=16.0,
        cbl_scale_distill_tiny_weight_cap=2.0,
        cbl_scale_distill_coordinate_reliable=True,
        cbl_scale_distill_stage={distill_stage!r},
        cbl_scale_distill_distance={distill_distance!r},
"""
        if is_distillation
        else ""
    )
    micro_rescue_args = (
        """        rpn_micro_rescue_teacher=teacher_path,
        rpn_micro_rescue_weight=0.005,
        rpn_micro_rescue_teacher_min_size=960,
        rpn_micro_rescue_teacher_max_size=1200,
        rpn_micro_rescue_proposal_top_n=300,
        rpn_micro_rescue_cutoff_px=8.0,
        rpn_micro_rescue_teacher_iou_floor=0.50,
        rpn_micro_rescue_margin=0.02,
"""
        if is_pc_micro_rescue
        else ""
    )
    micro_feature_args = (
        f"""        fpn_micro_feature_teacher=teacher_path,
        fpn_micro_feature_weight={micro_feature_weight!r},
        fpn_micro_feature_teacher_min_size=960,
        fpn_micro_feature_teacher_max_size=1200,
        fpn_micro_feature_proposal_top_n=300,
        fpn_micro_feature_cutoff_px=8.0,
        fpn_micro_feature_teacher_iou_floor=0.50,
        fpn_micro_feature_margin=0.02,
        fpn_micro_feature_target={micro_feature_target!r},
"""
        if is_pc_micro_feature
        else ""
    )
    return f"""from scripts.train_frcnn_metric import train_metric

{teacher_setup}
project_root = Path('/kaggle/working/cr_sc_cbl_project')
output_root = Path('/kaggle/working/tod_output')
output_root.mkdir(parents=True, exist_ok=True)
protocol = {{
    'method': {method_label!r},
    'method_key': {method!r},
    'source_bundle_sha256': SOURCE_BUNDLE_SHA256,
    'teacher_path': str(teacher_path) if teacher_path is not None else None,
    'teacher_sha256': teacher_sha256,
    'teacher_epoch': 5 if teacher_path is not None else None,
    'teacher_model_source': 'ema' if teacher_path is not None else None,
    'seed': {seed},
    'epochs': {epochs},
    'student_scale': [640, 800],
    'teacher_scale': [960, 1200] if teacher_path is not None else None,
    'distillation_weight': 0.25 if {is_distillation!r} else 0.0,
    'temperature': 2.0 if {is_distillation!r} else None,
    'distillation_stage': {distill_stage!r} if {is_distillation!r} else None,
    'distillation_distance': {distill_distance!r} if {is_distillation!r} else None,
    'rpn_micro_rescue_weight': 0.005 if {is_pc_micro_rescue!r} else 0.0,
    'rpn_micro_rescue_target': 'exact_gt_regression' if {is_pc_micro_rescue!r} else None,
    'rpn_micro_rescue_pcgrad_scope': 'student_rpn_head' if {is_pc_micro_rescue!r} else None,
    'rpn_micro_rescue_proposal_top_n': 300 if {is_pc_micro_rescue!r} else None,
    'rpn_micro_rescue_cutoff_px': 8.0 if {is_pc_micro_rescue!r} else None,
    'rpn_micro_rescue_teacher_iou_floor': 0.50 if {is_pc_micro_rescue!r} else None,
    'rpn_micro_rescue_margin': 0.02 if {is_pc_micro_rescue!r} else None,
    'fpn_micro_feature_weight': {micro_feature_weight!r} if {is_pc_micro_feature!r} else 0.0,
    'fpn_micro_feature_target': {micro_feature_target!r} if {is_pc_micro_feature!r} else None,
    'fpn_micro_feature_pcgrad_scope': 'student_fpn' if {is_pc_micro_feature!r} else None,
    'fpn_micro_feature_body_detached': True if {is_pc_micro_feature!r} else None,
    'fpn_micro_feature_proposal_top_n': 300 if {is_pc_micro_feature!r} else None,
    'fpn_micro_feature_cutoff_px': 8.0 if {is_pc_micro_feature!r} else None,
    'fpn_micro_feature_teacher_iou_floor': 0.50 if {is_pc_micro_feature!r} else None,
    'fpn_micro_feature_margin': 0.02 if {is_pc_micro_feature!r} else None,
    'selection': 'validation_mAP_50',
    'selected_checkpoint': 'best.pt',
    'locked_test_access': False,
}}
(output_root / 'protocol.json').write_text(json.dumps(protocol, indent=2))
(output_root / 'source_manifest.json').write_text(json.dumps(SOURCE_MANIFEST, indent=2))

try:
    train_metric(
        'sa_alw_full',
        'la_loss',
        {seed},
        box_loss='cbl',
        box_loss_warmup_epochs=0,
        tag={tag!r},
        cbl_refine_train_weight=0.5,
        cbl_refine_steps=1,
        cbl_refine_blend=1.0,
        cbl_refine_score_threshold=0.3,
{distillation_args}{micro_rescue_args}{micro_feature_args}        train_min_sizes=(640,),
        train_max_size=800,
    )
finally:
    run_root = project_root / 'runs'
    if run_root.exists():
        shutil.copytree(run_root, output_root / 'runs', dirs_exist_ok=True)
    print('artifact_copy_complete', flush=True)

print('done', {tag!r}, flush=True)
"""


def build_notebook(
    output: Path,
    *,
    method: str = "cr_sc_cbl",
    seed: int = 123,
    tag: str = "cr_sc_cbl_fair20",
    epochs: int = 20,
    use_ema: bool = True,
    teacher_sha256: str = TEACHER_SHA256,
) -> None:
    if method not in {
        "baseline", "cr_sc_cbl", "ra_cr_sc_cbl", "ra_tb_cbl",
        "pc_mr_rpn", "pc_moc_fd", "pc_mhfd", "ra_tb_pcmhfd",
        "pc_mr_moc"
    }:
        raise ValueError(f"Unsupported method: {method}")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if not tag or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in tag):
        raise ValueError("tag must contain only lowercase ASCII letters, digits, and underscores")
    if method != "baseline" and len(teacher_sha256) != 64:
        raise ValueError("teacher_sha256 must be a SHA-256 hex digest")
    manifest, bundle_digest = _source_manifest()
    method_labels = {
        "baseline": "Iterative-CBL baseline",
        "cr_sc_cbl": "CR-SC-CBL",
        "ra_cr_sc_cbl": "RA-CR-SC-CBL",
        "ra_tb_cbl": "RA-TB-CBL",
        "pc_mr_rpn": "PC-MR-RPN",
        "pc_moc_fd": "PC-MOC-FD",
        "pc_mhfd": "PC-MHFD",
        "ra_tb_pcmhfd": "RA-TB + PC-MHFD",
        "pc_mr_moc": "PC-MR-RPN + PC-MOC-FD",
    }
    method_label = method_labels[method]
    cells = [
        _cell(
            f"# {method_label} Fair-20 Seed {seed}\n\n"
            "Self-contained private training notebook. It consumes only the "
            "public TOD dataset"
            + (" and the frozen fair20 teacher." if method != "baseline" else ".")
            + " The locked test is not accessed.",
            cell_type="markdown",
        ),
        _cell("%pip install -q pycocotools torchmetrics\n"),
        _cell(
            "from __future__ import annotations\n"
            "import hashlib\n"
            "import json\n"
            "import os\n"
            "import shutil\n"
            "import sys\n"
            "import types\n"
            "from pathlib import Path\n"
            "\n"
            "os.environ.update({\n"
            "    'CPV_DATA_ROOT': '/kaggle/input/datasets/ngquangnht/tinydataset-yolostandard',\n"
            f"    'TOD_EPOCHS': '{epochs}',\n"
            "    'TOD_BATCH_SIZE': '4',\n"
            "    'TOD_NUM_WORKERS': '2',\n"
            f"    'TOD_USE_EMA': '{int(use_ema)}',\n"
            "    'TOD_USE_COPY_PASTE': '1',\n"
            "    'TOD_TINY_TILE_OVERSAMPLE': '2.0',\n"
            "    'TOD_EMPTY_CACHE_EVERY': '0',\n"
            "})\n"
            "\n"
            "import torch\n"
            "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\n"
            "gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]\n"
            "print('gpus', gpu_names)\n"
            "on_kaggle = bool(os.environ.get('KAGGLE_KERNEL_RUN_TYPE'))\n"
            "if on_kaggle and (len(gpu_names) != 2 or not all('T4' in name for name in gpu_names)):\n"
            "    raise RuntimeError(f'Expected exactly two Tesla T4 GPUs, got {gpu_names}')\n"
            "\n"
            "def _package(name: str):\n"
            "    module = types.ModuleType(name)\n"
            "    module.__path__ = []\n"
            "    module.__package__ = name\n"
            "    sys.modules[name] = module\n"
            "    return module\n"
            "\n"
            "def _exec_module(name: str, source: str, filename: str):\n"
            "    module = sys.modules.get(name) or types.ModuleType(name)\n"
            "    module.__file__ = filename\n"
            "    module.__package__ = name if hasattr(module, '__path__') else name.rpartition('.')[0]\n"
            "    sys.modules[name] = module\n"
            "    exec(compile(source, filename, 'exec'), module.__dict__)\n"
            "    return module\n"
            "\n"
            "_package('common')\n"
            "_package('common.metrics')\n"
            "_package('scripts')\n"
        ),
    ]
    cells.extend(_cell(_module_cell(name, path)) for name, path in MODULES)
    cells.append(
        _cell(
            f"SOURCE_MANIFEST = {json.dumps(manifest, indent=2)}\n"
            f"SOURCE_BUNDLE_SHA256 = {bundle_digest!r}\n"
            "print('source_bundle_sha256', SOURCE_BUNDLE_SHA256)\n"
        )
    )
    cells.append(
        _cell(_training_cell(
            method=method,
            seed=seed,
            tag=tag,
            epochs=epochs,
            manifest=manifest,
            bundle_digest=bundle_digest,
            teacher_sha256=teacher_sha256,
        ))
    )

    notebook = {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, indent=1), encoding="utf-8")

    parsed = json.loads(output.read_text(encoding="utf-8"))
    if parsed["nbformat"] != 4:
        raise RuntimeError("Notebook JSON validation failed")
    all_source = "\n".join(
        "".join(cell["source"]) for cell in parsed["cells"])
    for forbidden in ("sys.path.append", "sys.path.insert", "%run"):
        if forbidden in all_source:
            raise RuntimeError(f"Forbidden notebook dependency pattern: {forbidden}")
    for index, cell in enumerate(parsed["cells"]):
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if source.lstrip().startswith("%"):
            continue
        compile(source, f"notebook-cell-{index}", "exec")
    print(f"wrote {output}")
    print(f"cells={len(cells)} source_bundle_sha256={bundle_digest}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--method",
        choices=(
            "baseline", "cr_sc_cbl", "ra_cr_sc_cbl", "ra_tb_cbl",
            "pc_mr_rpn", "pc_moc_fd", "pc_mhfd", "ra_tb_pcmhfd",
            "pc_mr_moc"),
        default="cr_sc_cbl",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--tag", default="cr_sc_cbl_fair20")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument("--teacher-sha256", default=TEACHER_SHA256)
    args = parser.parse_args()
    build_notebook(
        args.output,
        method=args.method,
        seed=args.seed,
        tag=args.tag,
        epochs=args.epochs,
        use_ema=not args.no_ema,
        teacher_sha256=args.teacher_sha256,
    )


if __name__ == "__main__":
    main()
