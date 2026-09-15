from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fmgeo.forward import egg
from fmgeo.forward.egg import extract_egg_observations


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
