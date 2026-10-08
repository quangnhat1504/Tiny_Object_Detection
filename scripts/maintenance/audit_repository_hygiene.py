"""Audit and safely remove reproducible Tiny Object Detection build debris.

The default mode is read-only. ``--apply`` removes only allow-listed caches,
redundant checkpoints, and generated logs. Tracked files, sealed or private
data surfaces, runtime state, accepted evidence metadata, and every checkpoint
without a surviving selection checkpoint are protected.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import zipfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[2]
PROTECTED_TOP_LEVEL = {
    ".archive",
    ".git",
    ".runtime",
    ".worktrees",
    "data",
    "paper_a",
    "raw",
}
PROTECTED_PATHS = {
    Path("journal/test_raw"),
    Path("journal/audits"),
    Path("journal/results"),
    Path("journal/manuscript"),
    Path("journal/wiki"),
    Path("archive"),
    Path("reproducibility/H-WIoU"),
}
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


class UnverifiedCheckpointFormat(RuntimeError):
    """A checkpoint format that cannot be checked without unsafe deserialization."""


def _relative(root: Path, path: Path) -> Path:
    root = root.resolve()
    resolved = path.resolve()
    try:
        return resolved.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Refusing path outside repository: {resolved}") from error


def _is_link_or_reparse_point(path: Path) -> bool:
    try:
        file_stat = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    attributes = getattr(file_stat, "st_file_attributes", 0)
    return stat.S_ISLNK(file_stat.st_mode) or bool(attributes & reparse_flag)


def _has_link_or_reparse_component(root: Path, relative: Path) -> bool:
    current = root
    for part in relative.parts:
        if part in {"", "."}:
            continue
        current = current / part
        if _is_link_or_reparse_point(current):
            return True
    return False


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
    total = 0
    for current, directory_names, file_names in os.walk(
        path, topdown=True, followlinks=False
    ):
        current_path = Path(current)
        directory_names[:] = [
            name
            for name in directory_names
            if not _is_link_or_reparse_point(current_path / name)
        ]
        for name in file_names:
            item = current_path / name
            if _is_link_or_reparse_point(item):
                continue
            file_stat = item.lstat()
            if stat.S_ISREG(file_stat.st_mode):
                total += file_stat.st_size
    return total


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


def _is_git_repository_root(root: Path) -> bool:
    """Accept normal checkouts and linked worktrees, but not nested paths."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return Path(completed.stdout.strip()).resolve() == root.resolve()


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
        lexical_relative = path.relative_to(root)
        if _has_link_or_reparse_component(root, lexical_relative):
            return
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
            if _is_link_or_reparse_point(child):
                continue
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


def apply_cleanup(
    root: Path,
    candidates: list[CleanupCandidate],
    *,
    progress_callback: Callable[[int, dict[str, str]], None] | None = None,
) -> None:
    """Delete the pre-audited candidates after resolving every exact target."""
    root = root.resolve()

    def retry_readonly(function, path, _error) -> None:
        os.chmod(path, stat.S_IWRITE)
        function(path)

    for index, candidate in enumerate(candidates):
        if progress_callback is not None:
            progress_callback(index, {"status": "IN_PROGRESS"})
        relative = Path(candidate.path)
        try:
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(
                    f"Refusing non-relative cleanup target: {candidate.path}"
                )
            if candidate.kind not in {"directory", "file"}:
                raise RuntimeError(
                    f"Refusing unknown cleanup target kind: {candidate.kind}"
                )
            if _has_link_or_reparse_component(root, relative):
                raise RuntimeError(
                    f"Refusing symlink or reparse point cleanup target: {candidate.path}"
                )
            target = root / relative
            _relative(root, target)
            if _is_protected(root, target):
                raise RuntimeError(f"Protected path reached apply phase: {target}")
            if candidate.kind == "directory":
                if target.is_dir():
                    shutil.rmtree(target, onexc=retry_readonly)
                    outcome = {"status": "REMOVED"}
                elif target.exists():
                    raise RuntimeError(
                        f"Cleanup target changed kind: {candidate.path}"
                    )
                else:
                    outcome = {"status": "ALREADY_ABSENT"}
            elif target.is_file():
                try:
                    target.unlink()
                except PermissionError:
                    target.chmod(stat.S_IWRITE)
                    target.unlink()
                outcome = {"status": "REMOVED"}
            elif target.exists():
                raise RuntimeError(
                    f"Cleanup target changed kind: {candidate.path}"
                )
            else:
                outcome = {"status": "ALREADY_ABSENT"}
        except Exception as error:
            if progress_callback is not None:
                progress_callback(
                    index,
                    {"status": "FAILED", "error_type": type(error).__name__},
                )
            raise
        if progress_callback is not None:
            progress_callback(index, outcome)


def _resolve_report_target(root: Path, output: Path) -> Path:
    root = root.resolve()
    target = output if output.is_absolute() else root / output
    target = Path(os.path.abspath(target))
    try:
        relative = target.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"Refusing report outside repository: {target}") from error
    if _has_link_or_reparse_component(root, relative):
        raise RuntimeError(f"Refusing symlink or reparse point report path: {target}")
    if not relative.parts or (
        relative.parts[0] != ".runtime" and _is_protected(root, target)
    ):
        raise RuntimeError(f"Refusing report inside protected repository path: {target}")
    _relative(root, target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if _has_link_or_reparse_component(root, relative):
        raise RuntimeError(f"Refusing symlink or reparse point report path: {target}")
    return target


def _reserve_report_exclusive(
    root: Path, output: Path
) -> tuple[Path, str]:
    """Reserve a fresh report path before a destructive cleanup begins."""
    target = _resolve_report_target(root, output)
    run_id = uuid4().hex
    flags = os.O_CREAT | os.O_EXCL | os.O_RDWR
    if hasattr(os, "O_BINARY"):
        flags |= os.O_BINARY
    descriptor = os.open(target, flags, 0o600)
    pending = json.dumps(
        {
            "schema_version": 1,
            "scope": "local_repository_hygiene",
            "repository": str(root.resolve()),
            "mode": "APPLYING",
            "status": "IN_PROGRESS",
            "run_id": run_id,
        },
        indent=2,
        sort_keys=True,
    ) + "\n"
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(pending)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target, run_id


def _write_reserved_report(
    root: Path, target: Path, run_id: str, payload: dict
) -> None:
    """Atomically update the create-only report reserved by this run."""
    _resolve_report_target(root, target)
    if _is_link_or_reparse_point(target) or not target.is_file():
        raise RuntimeError("Reserved cleanup report was replaced or removed")
    try:
        current = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("Reserved cleanup report is unreadable") from error
    if current.get("run_id") != run_id:
        raise RuntimeError("Reserved cleanup report identity changed")

    payload["run_id"] = run_id
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        _resolve_report_target(root, target)
        if _is_link_or_reparse_point(target) or not target.is_file():
            raise RuntimeError("Reserved cleanup report was replaced or removed")
        current = json.loads(target.read_text(encoding="utf-8"))
        if current.get("run_id") != run_id:
            raise RuntimeError("Reserved cleanup report identity changed")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _report_overlaps_cleanup_candidates(
    root: Path,
    report_path: Path,
    candidates: list[CleanupCandidate],
) -> bool:
    report_relative = report_path.relative_to(root.resolve())
    return any(
        report_relative == Path(candidate.path)
        or _is_within(report_relative, Path(candidate.path))
        for candidate in candidates
    )


def _publish_report_exclusive(root: Path, output: Path, rendered: str) -> None:
    """Create a new in-repository report without following links or replacing files."""
    target = _resolve_report_target(root, output)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def verify_checkpoint_structure(path: Path) -> None:
    """Reject symlinks and check ZIP shape without loading its pickle payload."""
    if path.is_symlink():
        raise UnverifiedCheckpointFormat("Symlinked checkpoint is not inspected")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if not names or not any(
                name == "data.pkl" or name.endswith("/data.pkl")
                for name in names
            ):
                raise RuntimeError("PyTorch archive has no data.pkl")
        return

    raise UnverifiedCheckpointFormat(
        "Legacy non-zip checkpoint is not deserialized during cleanup"
    )


def prepare_cleanup_with_checkpoint_verification(
    root: Path,
    candidates: list[CleanupCandidate],
) -> tuple[list[CleanupCandidate], dict]:
    """Choose a structurally readable survivor and preserve unverified groups."""

    root = root.resolve()
    parents = sorted(
        {
            (root / item.path).parent
            for item in candidates
            if Path(item.path).suffix.lower() in CHECKPOINT_SUFFIXES
        }
    )
    adjusted = {item.path: item for item in candidates}
    checked = []
    corrupted = []
    unverified = []
    fallback_survivors = []
    for parent in parents:
        parent_relative = parent.relative_to(root)
        if _has_link_or_reparse_component(root, parent_relative):
            unverified.append(f"{parent_relative.as_posix()}::<link-parent>")
            choices = []
        else:
            epoch_paths = sorted(
                parent.glob("epoch_*.pt"),
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
            selected_relative = selected.relative_to(root)
            if _has_link_or_reparse_component(root, selected_relative):
                unverified.append(selected_relative.as_posix())
                continue
            if not selected.is_file() or selected.stat().st_size == 0:
                continue
            relative = _relative(root, selected).as_posix()
            try:
                verify_checkpoint_structure(selected)
            except UnverifiedCheckpointFormat:
                unverified.append(relative)
                continue
            except Exception:
                corrupted.append(relative)
                adjusted[relative] = CleanupCandidate(
                    path=relative,
                    kind="file",
                    size_bytes=selected.stat().st_size,
                    reason="corrupted checkpoint fails structural validation",
                )
                continue
            survivor = selected
            checked.append(relative)
            adjusted.pop(relative, None)
            if selected.name not in SELECTED_CHECKPOINT_NAMES:
                fallback_survivors.append(relative)
            break
        if survivor is None:
            relative_parent = parent_relative.as_posix()
            parent_relative = Path(relative_parent)
            for candidate_path in list(adjusted):
                candidate_relative = Path(candidate_path)
                if (
                    candidate_relative.parent == parent_relative
                    and candidate_relative.suffix.lower() in CHECKPOINT_SUFFIXES
                ):
                    adjusted.pop(candidate_path, None)
            fallback_survivors.append(f"NO_VERIFIED_SURVIVOR:{relative_parent}")
    status = "PASS_STRUCTURAL"
    if any(item.startswith("NO_VERIFIED_SURVIVOR:") for item in fallback_survivors):
        status = "HOLD_NO_VERIFIED_SURVIVOR"
    elif unverified:
        status = "PASS_STRUCTURAL_WITH_UNVERIFIED_CHECKPOINTS"
    elif corrupted:
        status = "PASS_STRUCTURAL_WITH_CORRUPTION_QUARANTINED"
    elif fallback_survivors:
        status = "PASS_STRUCTURAL_WITH_FALLBACK_SURVIVOR"
    return sorted(adjusted.values(), key=lambda item: item.path), {
        "status": status,
        "checkpoint_count": len(checked),
        "checkpoints": checked,
        "corrupted_checkpoint_count": len(corrupted),
        "corrupted_checkpoints": corrupted,
        "unverified_checkpoint_count": len(unverified),
        "unverified_checkpoints": unverified,
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
            "journal/audits/, journal/results/, journal/manuscript/, and journal/wiki/",
            "journal/test_raw/",
            "archive/ and reproducibility/H-WIoU/",
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
    if (
        not (root / "AGENT_HANDOVER.md").is_file()
        or not _is_git_repository_root(root)
    ):
        raise SystemExit("Refusing cleanup: repository identity markers are absent")
    if args.apply and not args.report:
        raise SystemExit("--apply requires a fresh --report path")

    report_path: Path | None = None
    report_run_id: str | None = None
    report_state: dict | None = None
    if args.apply:
        report_path, report_run_id = _reserve_report_exclusive(root, args.report)
        report_state = {
            "schema_version": 1,
            "scope": "local_repository_hygiene",
            "repository": str(root),
            "mode": "APPLYING",
            "status": "IN_PROGRESS",
            "cleanup_outcomes": [],
        }
    try:
        candidates = filter_cleanup_category(
            collect_cleanup_candidates(root), args.category
        )
        checkpoint_verification = None
        if args.apply:
            if report_path is None or _report_overlaps_cleanup_candidates(
                root, report_path, candidates
            ):
                raise RuntimeError(
                    "Refusing cleanup because the report is inside a candidate path"
                )
            candidates, checkpoint_verification = (
                prepare_cleanup_with_checkpoint_verification(root, candidates)
            )
            report_state = build_report(
                root,
                candidates,
                applied=True,
                surviving_checkpoint_verification=checkpoint_verification,
            )
            report_state.update(
                mode="APPLYING",
                status="IN_PROGRESS",
                cleanup_outcomes=[
                    {"path": candidate.path, "status": "PENDING"}
                    for candidate in candidates
                ],
            )
            if report_path is None or report_run_id is None:
                raise RuntimeError("Reserved cleanup report identity is missing")
            _write_reserved_report(root, report_path, report_run_id, report_state)

            def record_progress(index: int, update: dict[str, str]) -> None:
                if (
                    report_state is None
                    or report_path is None
                    or report_run_id is None
                ):
                    raise RuntimeError("Cleanup progress report is unavailable")
                report_state["cleanup_outcomes"][index].update(update)
                _write_reserved_report(
                    root, report_path, report_run_id, report_state
                )

            apply_cleanup(root, candidates, progress_callback=record_progress)
            unresolved = [
                outcome
                for outcome in report_state["cleanup_outcomes"]
                if outcome.get("status") not in {"REMOVED", "ALREADY_ABSENT"}
            ]
            if unresolved:
                raise RuntimeError(
                    "Cleanup returned with unresolved candidate outcomes"
                )
            report_state["mode"] = "APPLIED"
            report_state["status"] = "COMPLETE"
            _write_reserved_report(root, report_path, report_run_id, report_state)
            report = report_state
        else:
            report = build_report(root, candidates, applied=False)
        rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
        if not args.apply and args.report:
            _publish_report_exclusive(root, args.report, rendered)
        print(rendered, end="")
    except Exception as error:
        if (
            report_state is not None
            and report_path is not None
            and report_run_id is not None
        ):
            report_state.update(
                mode="APPLY_FAILED",
                status="INCOMPLETE",
                error_type=type(error).__name__,
            )
            try:
                _write_reserved_report(
                    root, report_path, report_run_id, report_state
                )
            except Exception:
                pass
        raise


if __name__ == "__main__":
    main()
