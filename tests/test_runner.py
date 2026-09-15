from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from fmgeo.forward.runner import DiskSpaceError, run_simulator


def write_script(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def extract_json(workdir: Path) -> dict[str, object]:
    return json.loads((workdir / "RESULT.json").read_text(encoding="utf-8"))


def test_disk_preflight_happens_before_process_launch(tmp_path: Path) -> None:
    marker = tmp_path / "launched"
    script = write_script(
        tmp_path / "sim.py",
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n",
    )

    with pytest.raises(DiskSpaceError, match="free disk"):
        run_simulator(
            [sys.executable, str(script)],
            work_root=tmp_path / "work",
            cache_dir=tmp_path / "cache",
            cache_key="a" * 64,
            extractor=extract_json,
            min_free_disk_gb=10**9,
        )

    assert not marker.exists()


def test_timeout_is_explicit_and_workdir_is_removed(tmp_path: Path) -> None:
    script = write_script(tmp_path / "sleep.py", "import time\ntime.sleep(5)\n")

    result = run_simulator(
        [sys.executable, str(script)],
        work_root=tmp_path / "work",
        cache_dir=tmp_path / "cache",
        cache_key="b" * 64,
        extractor=extract_json,
        timeout=0.05,
        min_free_disk_gb=0,
    )

    assert result.status == "timeout"
    assert result.d is None
    assert list((tmp_path / "work").iterdir()) == []


def test_failure_keeps_stderr_and_is_not_replaced(tmp_path: Path) -> None:
    script = write_script(
        tmp_path / "fail.py",
        "import sys\nprint('measured failure', file=sys.stderr)\nraise SystemExit(7)\n",
    )

    result = run_simulator(
        [sys.executable, str(script)],
        work_root=tmp_path / "work",
        cache_dir=tmp_path / "cache",
        cache_key="c" * 64,
        extractor=extract_json,
        min_free_disk_gb=0,
    )

    assert result.status == "failed"
    assert result.returncode == 7
    assert "measured failure" in result.stderr
    assert result.d is None


def test_success_is_cached_without_simulator_files(tmp_path: Path) -> None:
    counter = tmp_path / "counter"
    script = write_script(
        tmp_path / "success.py",
        "\n".join(
            [
                "import json",
                "from pathlib import Path",
                f"counter = Path({str(counter)!r})",
                "value = str(int(counter.read_text()) + 1) if counter.exists() else '1'",
                "counter.write_text(value)",
                "Path('RESULT.json').write_text(json.dumps({'d': [1.0, 2.0], 'FOPT_16.5y': 3.0}))",
            ]
        ),
    )
    kwargs = {
        "work_root": tmp_path / "work",
        "cache_dir": tmp_path / "cache",
        "cache_key": "d" * 64,
        "extractor": extract_json,
        "min_free_disk_gb": 0,
    }

    first = run_simulator([sys.executable, str(script)], **kwargs)
    second = run_simulator(["command-that-must-not-run"], **kwargs)

    assert first.status == "ok"
    assert first.d == (1.0, 2.0)
    assert first.fopt_16_5y == 3.0
    assert not first.cache_hit
    assert second.cache_hit
    assert counter.read_text() == "1"
    cache_files = list((tmp_path / "cache").iterdir())
    assert [path.suffix for path in cache_files] == [".json"]
    assert list((tmp_path / "work").iterdir()) == []


def test_prepare_callback_populates_isolated_workdir(tmp_path: Path) -> None:
    script = write_script(
        tmp_path / "prepared.py",
        "import json\nfrom pathlib import Path\n"
        "value = float(Path('INPUT').read_text())\n"
        "Path('RESULT.json').write_text(json.dumps({'d': [value], 'FOPT_16.5y': value}))\n",
    )

    result = run_simulator(
        [sys.executable, str(script)],
        work_root=tmp_path / "work",
        cache_dir=tmp_path / "cache",
        cache_key="e" * 64,
        extractor=extract_json,
        prepare=lambda workdir: (workdir / "INPUT").write_text("12.5"),
        min_free_disk_gb=0,
    )

    assert result.status == "ok"
    assert result.d == (12.5,)
