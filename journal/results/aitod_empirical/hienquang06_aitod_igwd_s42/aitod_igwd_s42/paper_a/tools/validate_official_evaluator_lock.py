from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = ROOT / "paper_a" / "evaluation" / "official_evaluator_lock.json"


def git_object(repo: Path, commit: str, path: str) -> bytes:
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{commit}:{path}"]
    )


def validate(sources_root: Path) -> list[dict[str, object]]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    results: list[dict[str, object]] = []
    for source in lock["sources"]:
        repo = sources_root / source["local_repo"]
        if not (repo / ".git").exists():
            raise FileNotFoundError(f"missing reference repository: {repo}")
        payload = git_object(repo, source["commit"], source["path"])
        actual = hashlib.sha256(payload).hexdigest()
        if actual != source["sha256"]:
            raise AssertionError(
                f"{source['name']} hash mismatch: {actual} != {source['sha256']}"
            )
        results.append(
            {
                "name": source["name"],
                "commit": source["commit"],
                "sha256": actual,
                "bytes": len(payload),
                "status": "PASS",
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources-root",
        type=Path,
        default=ROOT / ".runtime" / "paper_a_sources",
    )
    args = parser.parse_args()
    results = validate(args.sources_root.resolve())
    print(json.dumps({"status": "PASS", "sources": results}, indent=2))


if __name__ == "__main__":
    main()
