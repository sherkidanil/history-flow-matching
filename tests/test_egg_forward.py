from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from fmgeo.forward import egg
from fmgeo.forward.egg import extract_egg_observations, run_egg_forward
from fmgeo.forward.runner import ForwardResult


def test_extract_egg_observations_uses_exact_history_schedule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vectors = {
        "TIME": np.asarray([30.0, 60.0, 90.0, 120.0, 150.0]),
        "FOPT": np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        "WOPR:PROD1": np.asarray([10.0, 20.0, 30.0, 40.0, 50.0]),
        "WWPR:PROD1": np.asarray([1.0, 2.0, 3.0, 4.0, 5.0]),
        "WOPR:PROD2": np.asarray([15.0, 25.0, 35.0, 45.0, 55.0]),
        "WWPR:PROD2": np.asarray([1.5, 2.5, 3.5, 4.5, 5.5]),
    }
    monkeypatch.setattr(
        egg,
        "read_summary_vectors",
        lambda _case, keys: {key: vectors[key] for key in keys},
    )

    result = extract_egg_observations(
        tmp_path / "EGG",
        producers=("PROD1", "PROD2"),
        history_end_day=120.0,
        observation_interval_days=60.0,
        oil_rate_relative_sigma=0.05,
        water_rate_relative_sigma=0.10,
        rate_sigma_floor=1.0,
    )

    np.testing.assert_array_equal(result["times"], [60.0, 120.0])
    np.testing.assert_array_equal(result["d"], [20.0, 40.0, 25.0, 45.0, 2.0, 4.0, 2.5, 4.5])
    np.testing.assert_array_equal(result["sigma"], [1.0, 2.0, 1.25, 2.25, 1.0, 1.0, 1.0, 1.0])
    assert result["FOPT_16.5y"] == 5.0
    assert result["metadata"]["water_breakthrough_day"] == {
        "PROD1": 30.0,
        "PROD2": 30.0,
    }


def test_extract_egg_observations_rejects_nonmatching_schedule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vectors = {
        "TIME": np.asarray([30.0, 60.0, 95.0, 120.0]),
        "FOPT": np.ones(4),
        "WOPR:PROD1": np.ones(4),
        "WWPR:PROD1": np.ones(4),
    }
    monkeypatch.setattr(
        egg,
        "read_summary_vectors",
        lambda _case, keys: {key: vectors[key] for key in keys},
    )

    try:
        extract_egg_observations(
            tmp_path / "EGG",
            producers=("PROD1",),
            history_end_day=120.0,
            observation_interval_days=30.0,
            oil_rate_relative_sigma=0.05,
            water_rate_relative_sigma=0.05,
            rate_sigma_floor=1.0,
        )
    except ValueError as error:
        assert "schedule" in str(error)
    else:
        raise AssertionError("nonmatching schedule was accepted")


def test_run_egg_forward_prepares_permeability_and_caches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    template = tmp_path / "template"
    template.mkdir()
    (template / "EGG.DATA").write_text("RUNSPEC\n", encoding="utf-8")
    simulator = tmp_path / "simulate.py"
    simulator.write_text(
        "from pathlib import Path\n"
        "import json\n"
        "text = Path('mDARCY.INC').read_text()\n"
        "Path('RESULT.json').write_text(json.dumps({'has_permx': 'PERMX' in text}))\n",
        encoding="utf-8",
    )

    def fake_extract(case_path: str | Path, **_kwargs: object) -> dict[str, object]:
        payload = json.loads(Path(case_path).with_name("RESULT.json").read_text())
        assert payload["has_permx"]
        return {"d": [1.0, 2.0], "FOPT_16.5y": 3.0}

    monkeypatch.setattr(egg, "extract_egg_observations", fake_extract)
    def run_once() -> ForwardResult:
        return run_egg_forward(
            np.zeros((7, 60, 60)),
            template_dir=template,
            simulator_command=(sys.executable, str(simulator)),
            simulator_id="test-simulator-v1",
            work_root=tmp_path / "work",
            cache_dir=tmp_path / "cache",
            history_end_day=120.0,
            observation_interval_days=60.0,
            oil_rate_relative_sigma=0.05,
            water_rate_relative_sigma=0.05,
            rate_sigma_floor=1.0,
            min_free_disk_gb=0.0,
        )

    first = run_once()
    second = run_once()

    assert first.status == "ok"
    assert second.cache_hit
    assert len(list((tmp_path / "cache").glob("*.json"))) == 1
