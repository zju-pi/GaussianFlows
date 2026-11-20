"""Small runtime utilities shared by dataset adapters."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


def parse_seed_list(value: str) -> list[int]:
    """Parse comma-separated seeds and inclusive ranges such as ``0-199,250``."""
    seeds: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start_text, end_text = item.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise argparse.ArgumentTypeError(f"invalid seed range: {item}")
            seeds.extend(range(start, end + 1))
        else:
            seeds.append(int(item))
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be unique nonnegative integers")
    return seeds


def source_revision(source_repo: Path) -> str | None:
    """Return a Git revision, or the revision recorded with vendored source."""
    # A vendored directory lives inside the project's Git tree.  Reading its
    # marker first avoids accidentally reporting the enclosing project's HEAD.
    revision = source_repo / "SOURCE_REVISION"
    if revision.is_file():
        return revision.read_text(encoding="utf-8").strip()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source_repo,
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def file_fingerprint(path: Path) -> dict[str, int | str]:
    """Record file identity without hashing an entire large checkpoint."""
    path = path.expanduser().resolve()
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "first_mib_sha256": digest.hexdigest(),
    }
