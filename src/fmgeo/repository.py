"""Checks that private context and credentials stay outside Git history."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

PROHIBITED_FILES = {".env", "SOUL.md", "repo.md"}
PROHIBITED_PREFIXES = ("MEMORY/", "REPORTS/", "TASKS/", "docs/plans/")


def _git_lines(repo: Path, *args: str) -> list[str]:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in completed.stdout.splitlines() if line]


def find_tracked_prohibited(repo: Path) -> list[str]:
    """Return tracked or staged paths that cross the publication boundary."""
    paths = _git_lines(repo, "ls-files")
    return sorted(
        path
        for path in paths
        if path in PROHIBITED_FILES or path.startswith(PROHIBITED_PREFIXES)
    )


def find_staged_key_names(repo: Path, secret_keys: set[str]) -> list[str]:
    """Find staged text lines that assign known secret variable names."""
    if not secret_keys:
        return []
    patterns = {
        key: re.compile(rf"^\s*(?:export\s+)?{re.escape(key)}\s*[:=]", re.MULTILINE)
        for key in secret_keys
    }
    paths = _git_lines(repo, "diff", "--cached", "--name-only", "--diff-filter=ACMR")
    violations: list[str] = []
    for relative_path in paths:
        try:
            text = (repo / relative_path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        violations.extend(
            f"{relative_path}:{key}" for key, pattern in patterns.items() if pattern.search(text)
        )
    return sorted(violations)

