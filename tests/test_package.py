from __future__ import annotations

import subprocess
import sys


def test_package_exposes_version() -> None:
    import fmgeo

    assert fmgeo.__version__


def test_cli_help_does_not_require_optional_runtimes() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "fmgeo.cli", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Flow-matching geological inversion" in completed.stdout

