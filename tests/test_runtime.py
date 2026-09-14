from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from fmgeo.runtime import probe_command, probe_runtime, select_device


@pytest.mark.parametrize(
    ("requested", "cuda", "mps", "expected"),
    [
        ("cpu", True, True, "cpu"),
        ("cuda", True, True, "cuda"),
        ("mps", True, True, "mps"),
        ("auto", True, True, "cuda"),
        ("auto", False, True, "mps"),
        ("auto", False, False, "cpu"),
    ],
)
def test_device_selection_order(
    requested: str, cuda: bool, mps: bool, expected: str
) -> None:
    assert select_device(requested, cuda_available=cuda, mps_available=mps) == expected


def test_explicit_unavailable_accelerator_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="CUDA"):
        select_device("cuda", cuda_available=False, mps_available=True)


def test_missing_command_is_structured_not_exception() -> None:
    result = probe_command("fmgeo-command-that-does-not-exist")

    assert result == {"available": False, "path": None, "version": None}


def test_runtime_probe_has_portable_resource_fields(tmp_path: Path) -> None:
    result = probe_runtime(tmp_path, flow_command="fmgeo-command-that-does-not-exist")

    assert result["platform"]["system"]
    assert result["platform"]["python"]
    assert result["resources"]["cpu_count"] >= 1
    assert result["resources"]["memory_bytes"] > 0
    assert result["resources"]["disk_free_bytes"] > 0
    assert result["opm"]["available"] is False
    assert result["scheduler"]["type"] in {"none", "slurm"}


def test_probe_only_script_prints_json() -> None:
    completed = subprocess.run(
        [sys.executable, "scripts/m0_smoke_test.py", "--probe-only"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["platform"]["system"]


def test_probe_can_write_measured_environment_snapshot(tmp_path: Path) -> None:
    destination = tmp_path / "environment.md"
    completed = subprocess.run(
        [
            sys.executable,
            "scripts/m0_smoke_test.py",
            "--probe-only",
            "--environment-file",
            str(destination),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    recorded = destination.read_text(encoding="utf-8")
    payload = json.loads(completed.stdout)
    assert f"- Operating system: {payload['platform']['system']}" in recorded
    assert f"- CPU count: {payload['resources']['cpu_count']}" in recorded
