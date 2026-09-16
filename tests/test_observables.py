from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fmgeo.forward import observables
from fmgeo.forward.observables import (
    ObservationErrorConfig,
    observation_sigmas,
    read_summary_vectors,
)


def test_observation_sigmas_switch_on_shutin_and_breakthrough() -> None:
    config = ObservationErrorConfig(
        shutin_pressure_bar=1.0,
        flowing_pressure_bar=3.0,
        gor_pre_fraction=0.10,
        gor_post_fraction=0.25,
        wct_pre_absolute=0.02,
        wct_post_absolute=0.05,
        gas_breakthrough_threshold=10.0,
        water_breakthrough_threshold=0.05,
        zero_rate_threshold=1e-8,
    )

    result = observation_sigmas(
        bhp=np.array([200.0, 210.0, 220.0]),
        oil_rate=np.array([100.0, 0.0, 50.0]),
        gor=np.array([5.0, 20.0, 30.0]),
        wct=np.array([0.01, 0.10, 0.20]),
        config=config,
    )

    np.testing.assert_array_equal(result["bhp"], [3.0, 1.0, 3.0])
    np.testing.assert_allclose(result["gor"], [0.5, 5.0, 7.5])
    np.testing.assert_array_equal(result["wct"], [0.02, 0.05, 0.05])


def test_breakthrough_state_is_sticky_after_first_crossing() -> None:
    config = ObservationErrorConfig(gas_breakthrough_threshold=10.0)

    result = observation_sigmas(
        bhp=np.ones(3),
        oil_rate=np.ones(3),
        gor=np.array([5.0, 20.0, 5.0]),
        wct=np.zeros(3),
        config=config,
    )

    np.testing.assert_allclose(result["gor"], [0.5, 5.0, 1.25])


def test_read_summary_vectors_uses_resdata_and_returns_float_arrays(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeSummary:
        def __contains__(self, key: str) -> bool:
            return key in {"TIME", "FOPT"}

        def numpy_vector(self, key: str) -> np.ndarray:
            values = {"TIME": [0, 1], "FOPT": [0, 12.5]}
            return np.asarray(values[key])

    monkeypatch.setattr(observables, "_load_resdata_summary", lambda _: FakeSummary())

    result = read_summary_vectors(tmp_path / "SPE1CASE1", ["TIME", "FOPT"])

    assert result["TIME"].dtype == np.float64
    assert result["FOPT"].tolist() == [0.0, 12.5]


def test_read_summary_vectors_rejects_missing_or_misaligned_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeSummary:
        def __contains__(self, key: str) -> bool:
            return key != "MISSING"

        def numpy_vector(self, key: str) -> np.ndarray:
            return np.asarray([1.0] if key == "SHORT" else [1.0, 2.0])

    monkeypatch.setattr(observables, "_load_resdata_summary", lambda _: FakeSummary())

    with pytest.raises(KeyError, match="MISSING"):
        read_summary_vectors(tmp_path / "CASE", ["FOPT", "MISSING"])
    with pytest.raises(ValueError, match="identical lengths"):
        read_summary_vectors(tmp_path / "CASE", ["FOPT", "SHORT"])


def test_read_summary_vectors_allows_resdata_virtual_time_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeSummary:
        def __contains__(self, key: str) -> bool:
            return False

        def numpy_vector(self, key: str) -> np.ndarray:
            if key != "TIME":
                raise KeyError(key)
            return np.asarray([30.0, 60.0])

    monkeypatch.setattr(observables, "_load_resdata_summary", lambda _: FakeSummary())

    result = read_summary_vectors(tmp_path / "CASE", ["TIME"])

    np.testing.assert_array_equal(result["TIME"], [30.0, 60.0])
