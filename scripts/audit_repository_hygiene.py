"""Audit and safely remove reproducible Tiny Object Detection build debris.

The default mode is read-only. ``--apply`` removes only allow-listed caches,
redundant checkpoints, and generated logs. Tracked files, sealed or private
data surfaces, runtime state, accepted evidence metadata, and every checkpoint
without a surviving selection checkpoint are protected.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import stat
import subprocess
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTECTED_TOP_LEVEL = {
    ".archive",
    ".git",
    ".runtime",
    ".worktrees",
    "data",
    "paper_a",
    "raw",
}
PROTECTED_PATHS = {Path("journal/test_raw")}
SELECTED_CHECKPOINT_NAMES = (
    "best.pt",
    "best_ap75.pt",
    "best_coco_ap.pt",
    "best_checkpoint_state.pt",
)
CHECKPOINT_SUFFIXES = {".pt", ".pth", ".ckpt"}
BACKBONE_CACHE_NAME = "fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"
EPOCH_CHECKPOINT = re.compile(r"^epoch_\d+\.pt$")
CLEANUP_CATEGORIES = {"all", "checkpoints", "caches", "logs"}


@dataclass(frozen=True)
class CleanupCandidate:
    path: str
    kind: str
    size_bytes: int
    reason: str


def _relative(root: Path, path: Path) -> Path:
    root = root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Refusing path outside repository: {resolved}") from error


def _is_within(relative: Path, protected: Path) -> bool:
    return relative == protected or protected in relative.parents


def _is_protected(root: Path, path: Path) -> bool:
    relative = _relative(root, path)
    if not relative.parts:
        return True
    if relative.parts[0] in PROTECTED_TOP_LEVEL or relative.parts[0].startswith(
        ".venv"
    ):
        return True
    return any(_is_within(relative, protected) for protected in PROTECTED_PATHS)


def _directory_size(path: Path) -> int:
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file()
    )


def _tracked_paths(root: Path) -> set[str]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return {
        item.decode("utf-8", errors="surrogateescape").replace("\\", "/")
        for item in completed.stdout.split(b"\0")
        if item
    }


def _is_tracked(relative: Path, tracked: set[str]) -> bool:
    key = relative.as_posix()
    return key in tracked or any(item.startswith(key + "/") for item in tracked)


def _has_selected_sibling(path: Path) -> bool:
    return any((path.parent / name).is_file() for name in SELECTED_CHECKPOINT_NAMES)


def collect_cleanup_candidates(
    root: Path = ROOT,
    *,
    tracked: set[str] | None = None,
) -> list[CleanupCandidate]:
    """Return a deterministic allow-list of safe cleanup targets."""
    root = root.resolve()
    tracked = _tracked_paths(root) if tracked is None else tracked
    candidates: dict[Path, CleanupCandidate] = {}

    def add(path: Path, kind: str, reason: str) -> None:
        relative = _relative(root, path)
        if _is_protected(root, path) or _is_tracked(relative, tracked):
            return
        size = _directory_size(path) if path.is_dir() else path.stat().st_size
        candidates[relative] = CleanupCandidate(
            path=relative.as_posix(),
            kind=kind,
            size_bytes=size,
            reason=reason,
        )

    pytest_cache = root / ".pytest_cache"
    if pytest_cache.is_dir():
        add(pytest_cache, "directory", "reproducible pytest cache")

    for current, directory_names, file_names in os.walk(root, topdown=True):
        current_path = Path(current)
        current_relative = current_path.relative_to(root)
        retained_directories = []
        for name in directory_names:
            child = current_path / name
            child_relative = current_relative / name
            if not current_relative.parts and (
                name in PROTECTED_TOP_LEVEL or name.startswith(".venv")
            ):
                continue
            if any(
                _is_within(child_relative, protected)
                for protected in PROTECTED_PATHS
            ):
                continue
            if name == "__pycache__":
                add(child, "directory", "reproducible Python bytecode cache")
                continue
            retained_directories.append(name)
        directory_names[:] = retained_directories

        for name in file_names:
            path = current_path / name
            if path.suffix == ".pyc":
                add(path, "file", "reproducible Python bytecode")
            elif path.name == BACKBONE_CACHE_NAME and tuple(
                part.lower() for part in path.parts[-4:-1]
            ) == ("torch_cache", "hub", "checkpoints"):
                add(
                    path,
                    "file",
                    "duplicated downloadable torchvision backbone cache",
                )
            elif (
                path.suffix.lower() in CHECKPOINT_SUFFIXES
                and path.stat().st_size == 0
            ):
                add(path, "file", "zero-byte corrupted checkpoint")
            elif path.name == "last.pt" and _has_selected_sibling(path):
                add(
                    path,
                    "file",
                    "redundant final state; a selected checkpoint survives beside it",
                )
            elif EPOCH_CHECKPOINT.fullmatch(path.name) and _has_selected_sibling(path):
                add(
                    path,
                    "file",
                    "redundant epoch snapshot; a selected checkpoint survives beside it",
                )
            elif path.name.endswith((".pt.tmp", ".pth.tmp", ".ckpt.tmp")):
                add(path, "file", "interrupted atomic checkpoint temporary file")
            elif path.stat().st_size == 0 and path.suffix.lower() in {
                ".log",
                ".err",
                ".out",
            }:
                add(path, "file", "empty generated log")

    directory_paths = {
        Path(candidate.path)
        for candidate in candidates.values()
        if candidate.kind == "directory"
    }
    filtered = [
        candidate
        for relative, candidate in candidates.items()
        if not any(parent in relative.parents for parent in directory_paths)
    ]
    return sorted(filtered, key=lambda item: item.path)


def filter_cleanup_category(
    candidates: list[CleanupCandidate], category: str
) -> list[CleanupCandidate]:
    """Select an operational chunk without broadening the cleanup allow-list."""
    if category == "all":
        return candidates
    if category == "checkpoints":
        markers = ("checkpoint", "epoch snapshot", "final state")
    elif category == "caches":
        markers = ("cache", "bytecode")
    elif category == "logs":
        markers = ("log",)
    else:
        raise ValueError(f"Unknown cleanup category: {category}")
    return [
        item for item in candidates
        if any(marker in item.reason for marker in markers)
    ]


def apply_cleanup(root: Path, candidates: list[CleanupCandidate]) -> None:
    """Delete the pre-audited candidates after resolving every exact target."""
    root = root.resolve()

    def retry_readonly(function, path, _error) -> None:
        os.chmod(path, stat.S_IWRITE)
        function(path)

    for candidate in candidates:
        target = (root / candidate.path).resolve()
        _relative(root, target)
        if _is_protected(root, target):
            raise RuntimeError(f"Protected path reached apply phase: {target}")
        if candidate.kind == "directory":
            if target.is_dir():
                shutil.rmtree(target, onexc=retry_readonly)
        elif target.is_file():
            try:
                target.unlink()
            except PermissionError:
                target.chmod(stat.S_IWRITE)
                target.unlink()


def prepare_cleanup_with_checkpoint_verification(
    root: Path,
    candidates: list[CleanupCandidate],
) -> tuple[list[CleanupCandidate], dict]:
    """Choose a structurally readable survivor and classify corrupt files."""

    def verify_checkpoint(path: Path) -> None:
        if zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                if not names or not any(
                    name == "data.pkl" or name.endswith("/data.pkl")
                    for name in names
                ):
                    raise RuntimeError("PyTorch archive has no data.pkl")
            return

        # PyTorch used a raw-pickle serialization format before zip archives.
        # Only legacy files take this slower compatibility path.
        import torch

        payload = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not payload:
            raise RuntimeError("checkpoint payload is empty")
        model_state = payload.get("model_state_dict", payload)
        if not isinstance(model_state, dict) or not model_state:
            raise RuntimeError("checkpoint has no model state")
        del payload
        gc.collect()

    root = root.resolve()
    reasons = {
        "redundant final state; a selected checkpoint survives beside it",
        "redundant epoch snapshot; a selected checkpoint survives beside it",
    }
    parents = sorted({
        (root / item.path).parent
        for item in candidates
        if item.reason in reasons
    })
    adjusted = {item.path: item for item in candidates}
    checked = []
    corrupted = []
    fallback_survivors = []
    for parent in parents:
        epoch_paths = sorted(
            (path for path in parent.glob("epoch_*.pt") if path.is_file()),
            key=lambda path: int(path.stem.split("_", 1)[1]),
            reverse=True,
        )
        choices = [
            *(parent / name for name in SELECTED_CHECKPOINT_NAMES),
            parent / "last.pt",
            *epoch_paths,
        ]
        survivor = None
        for selected in choices:
            if not selected.is_file() or selected.stat().st_size == 0:
                continue
            relative = _relative(root, selected).as_posix()
            try:
                verify_checkpoint(selected)
            except Exception:
                corrupted.append(relative)
                adjusted[relative] = CleanupCandidate(
                    path=relative,
                    kind="file",
                    size_bytes=selected.stat().st_size,
                    reason="corrupted checkpoint fails torch.load",
                )
                continue
            survivor = selected
            checked.append(relative)
            adjusted.pop(relative, None)
            if selected.name not in SELECTED_CHECKPOINT_NAMES:
                fallback_survivors.append(relative)
            break
        if survivor is None:
            relative_parent = _relative(root, parent).as_posix()
            fallback_survivors.append(f"NO_LOADABLE_SURVIVOR:{relative_parent}")
    status = "PASS_STRUCTURAL"
    if corrupted or fallback_survivors:
        status = "PASS_STRUCTURAL_WITH_CORRUPTION_QUARANTINED"
    return sorted(adjusted.values(), key=lambda item: item.path), {
        "status": status,
        "checkpoint_count": len(checked),
        "checkpoints": checked,
        "corrupted_checkpoint_count": len(corrupted),
        "corrupted_checkpoints": corrupted,
        "fallback_survivors": fallback_survivors,
    }


def build_report(
    root: Path,
    candidates: list[CleanupCandidate],
    *,
    applied: bool,
    surviving_checkpoint_verification: dict | None = None,
) -> dict:
    by_reason = Counter(item.reason for item in candidates)
    bytes_by_reason = Counter()
    for item in candidates:
        bytes_by_reason[item.reason] += item.size_bytes
    return {
        "schema_version": 1,
        "scope": "local_repository_hygiene",
        "seed_policy": "SEED_42_ONLY",
        "official_final_test": "CLOSED",
        "repository": str(root.resolve()),
        "mode": "APPLIED" if applied else "DRY_RUN",
        "candidate_count": len(candidates),
        "candidate_bytes": sum(item.size_bytes for item in candidates),
        "protected": [
            "Git-tracked files",
            "data/, raw/, paper_a/, .runtime/, and .worktrees/",
            "journal/test_raw/",
            ".archive/",
            "checkpoints without a surviving selected checkpoint",
        ],
        "counts_by_reason": dict(sorted(by_reason.items())),
        "bytes_by_reason": dict(sorted(bytes_by_reason.items())),
        "surviving_checkpoint_verification": (
            surviving_checkpoint_verification
            if surviving_checkpoint_verification is not None
            else {"status": "NOT_RUN"}
        ),
        "candidates": [asdict(item) for item in candidates],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--category", choices=sorted(CLEANUP_CATEGORIES), default="all"
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not (root / "AGENT_HANDOVER.md").is_file() or not (root / ".git").is_dir():
        raise SystemExit("Refusing cleanup: repository identity markers are absent")
    candidates = filter_cleanup_category(
        collect_cleanup_candidates(root), args.category
    )
    checkpoint_verification = None
    if args.apply:
        candidates, checkpoint_verification = (
            prepare_cleanup_with_checkpoint_verification(root, candidates)
        )
        apply_cleanup(root, candidates)
    report = build_report(
        root,
        candidates,
        applied=args.apply,
        surviving_checkpoint_verification=checkpoint_verification,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report:
        output = args.report.resolve()
        _relative(root, output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
