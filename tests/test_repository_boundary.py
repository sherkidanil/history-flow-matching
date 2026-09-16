from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

from fmgeo.repository import find_staged_key_names, find_tracked_prohibited


def git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


def init_repo(path: Path) -> None:
    git(path, "init", "-q")
    git(path, "config", "user.name", "Test User")
    git(path, "config", "user.email", "test@example.invalid")


def test_tracked_private_paths_are_reported(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / ".env").write_text("placeholder\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("public\n", encoding="utf-8")
    git(tmp_path, "add", "-f", ".env")
    git(tmp_path, "add", "README.md")

    assert find_tracked_prohibited(tmp_path) == [".env"]


def test_staged_secret_key_names_are_reported(tmp_path: Path) -> None:
    init_repo(tmp_path)
    (tmp_path / "settings.txt").write_text(
        "TOP_SECRET_TOKEN=do-not-commit\n", encoding="utf-8"
    )
    git(tmp_path, "add", "settings.txt")

    assert find_staged_key_names(tmp_path, {"TOP_SECRET_TOKEN"}) == [
        "settings.txt:TOP_SECRET_TOKEN"
    ]


def test_current_repository_tracks_no_private_paths() -> None:
    assert find_tracked_prohibited(Path.cwd()) == []


def test_ci_covers_linux_and_macos_with_locked_uv() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    matrix = workflow["jobs"]["test"]["strategy"]["matrix"]["os"]
    steps = workflow["jobs"]["test"]["steps"]

    assert matrix == ["ubuntu-latest", "macos-latest"]
    assert any(step.get("run") == "uv sync --locked" for step in steps)

