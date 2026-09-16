"""Observation error rules for PUNQ-S3 history matching."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray


class _SummaryReader(Protocol):
    def __contains__(self, key: object) -> bool: ...

    def numpy_vector(self, key: str) -> ArrayLike: ...


def _load_resdata_summary(case_path: Path) -> _SummaryReader:
    module = importlib.import_module("resdata.summary")
    summary_type = module.Summary
    return cast(_SummaryReader, summary_type(str(case_path)))


def read_summary_vectors(
    case_path: str | Path, keys: list[str] | tuple[str, ...]
) -> dict[str, NDArray[np.float64]]:
    """Read finite, aligned summary vectors with the pinned ResData dependency."""
    if not keys or any(not key for key in keys):
        raise ValueError("at least one non-empty summary key is required")
    summary = _load_resdata_summary(Path(case_path))
    vectors: dict[str, NDArray[np.float64]] = {}
    expected_length: int | None = None
    for key in keys:
        # ResData exposes TIME as a computed vector but does not list it as a
        # regular summary keyword on every supported release.
        if key != "TIME" and key not in summary:
            raise KeyError(f"summary key not found: {key}")
        values = np.asarray(summary.numpy_vector(key), dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError(f"summary vector {key} must be finite and one-dimensional")
        if expected_length is None:
            expected_length = len(values)
        elif len(values) != expected_length:
            raise ValueError("summary vectors must have identical lengths")
        vectors[key] = values
    return vectors


@dataclass(frozen=True)
class ObservationErrorConfig:
    """Benchmark error magnitudes and state thresholds."""

    shutin_pressure_bar: float = 1.0
    flowing_pressure_bar: float = 3.0
    gor_pre_fraction: float = 0.10
    gor_post_fraction: float = 0.25
    wct_pre_absolute: float = 0.02
    wct_post_absolute: float = 0.05
    gas_breakthrough_threshold: float = 10.0
    water_breakthrough_threshold: float = 0.05
    zero_rate_threshold: float = 1e-8


def _sticky_breakthrough(values: NDArray[np.float64], threshold: float) -> NDArray[np.bool_]:
    return np.maximum.accumulate(values >= threshold)


def observation_sigmas(
    *,
    bhp: ArrayLike,
    oil_rate: ArrayLike,
    gor: ArrayLike,
    wct: ArrayLike,
    config: ObservationErrorConfig,
) -> dict[str, NDArray[np.float64]]:
    """Apply shut-in and sticky breakthrough rules to one well time series."""
    pressure = np.asarray(bhp, dtype=np.float64)
    rate = np.asarray(oil_rate, dtype=np.float64)
    gas_ratio = np.asarray(gor, dtype=np.float64)
    water_cut = np.asarray(wct, dtype=np.float64)
    if not (pressure.shape == rate.shape == gas_ratio.shape == water_cut.shape):
        raise ValueError("all well time series must have identical shapes")
    if pressure.ndim != 1 or not all(
        np.all(np.isfinite(values)) for values in (pressure, rate, gas_ratio, water_cut)
    ):
        raise ValueError("well time series must be finite one-dimensional arrays")

    shut_in = np.abs(rate) <= config.zero_rate_threshold
    gas_after = _sticky_breakthrough(gas_ratio, config.gas_breakthrough_threshold)
    water_after = _sticky_breakthrough(water_cut, config.water_breakthrough_threshold)
    return {
        "bhp": np.where(
            shut_in, config.shutin_pressure_bar, config.flowing_pressure_bar
        ).astype(np.float64),
        "gor": gas_ratio
        * np.where(gas_after, config.gor_post_fraction, config.gor_pre_fraction),
        "wct": np.where(
            water_after, config.wct_post_absolute, config.wct_pre_absolute
        ).astype(np.float64),
    }
