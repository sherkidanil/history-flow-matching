from __future__ import annotations

from pathlib import Path

import pytest

from fmgeo.config import load_config


def write_config(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def valid_config() -> str:
    return """
seed: 17
scale: smoke
execution:
  profile: local
  workers: 2
  device: auto
  min_free_disk_gb: 20
  work_dir: run/scratch
  flow_command: flow
esmda:
  ensemble_size: 100
  inflations: [4.0, 4.0, 4.0, 4.0]
"""


def test_relative_paths_resolve_from_config_directory(tmp_path: Path) -> None:
    config_path = write_config(tmp_path / "experiment.yaml", valid_config())

    config = load_config(config_path)

    assert config.execution.work_dir == (tmp_path / "run/scratch").resolve()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("seed: 17", "seed: -1"),
        ("min_free_disk_gb: 20", "min_free_disk_gb: -1"),
        ("inflations: [4.0, 4.0, 4.0, 4.0]", "inflations: [2.0, 4.0]"),
    ],
)
def test_invalid_numerical_configuration_is_rejected(
    tmp_path: Path, old: str, new: str
) -> None:
    config_path = write_config(tmp_path / "invalid.yaml", valid_config().replace(old, new))

    with pytest.raises(ValueError):
        load_config(config_path)


def test_unknown_keys_are_rejected(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path / "unknown.yaml", valid_config() + "unexpected: true\n"
    )

    with pytest.raises(ValueError, match="unexpected"):
        load_config(config_path)
