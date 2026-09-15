"""Egg-specific summary extraction for history matching."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fmgeo.forward.observables import read_summary_vectors

EGG_PRODUCERS = ("PROD1", "PROD2", "PROD3", "PROD4")


def _history_indices(
    times: np.ndarray, *, history_end_day: float, observation_interval_days: float
) -> np.ndarray:
    if history_end_day <= 0 or observation_interval_days <= 0:
        raise ValueError("history end and observation interval must be positive")
    count = history_end_day / observation_interval_days
    if not np.isclose(count, round(count), rtol=0.0, atol=1e-10):
        raise ValueError("history end must be divisible by the observation interval")
    requested = np.arange(1, int(round(count)) + 1, dtype=np.float64)
    requested *= observation_interval_days
    indices: list[int] = []
    for target in requested:
        matches = np.flatnonzero(np.isclose(times, target, rtol=0.0, atol=1e-8))
        if len(matches) != 1:
            raise ValueError(f"summary schedule has no unique output at day {target:g}")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=np.int64)


def extract_egg_observations(
    case_path: str | Path,
    *,
    producers: Sequence[str] = EGG_PRODUCERS,
    history_end_day: float,
    observation_interval_days: float,
    oil_rate_relative_sigma: float,
    water_rate_relative_sigma: float,
    rate_sigma_floor: float,
) -> dict[str, Any]:
    """Extract rate observations and their declared independent errors."""

    if not producers or any(not producer for producer in producers):
        raise ValueError("at least one producer is required")
    if oil_rate_relative_sigma <= 0 or water_rate_relative_sigma <= 0:
        raise ValueError("relative rate errors must be positive")
    if rate_sigma_floor <= 0:
        raise ValueError("rate error floor must be positive")
    oil_keys = [f"WOPR:{producer}" for producer in producers]
    water_keys = [f"WWPR:{producer}" for producer in producers]
    keys = ["TIME", "FOPT", *oil_keys, *water_keys]
    vectors = read_summary_vectors(case_path, keys)
    times = vectors["TIME"]
    indices = _history_indices(
        times,
        history_end_day=history_end_day,
        observation_interval_days=observation_interval_days,
    )
    oil = np.concatenate([vectors[key][indices] for key in oil_keys])
    water = np.concatenate([vectors[key][indices] for key in water_keys])
    observations = np.concatenate([oil, water])
    sigma = np.concatenate(
        [
            np.maximum(np.abs(oil) * oil_rate_relative_sigma, rate_sigma_floor),
            np.maximum(np.abs(water) * water_rate_relative_sigma, rate_sigma_floor),
        ]
    )
    return {
        "d": observations,
        "sigma": sigma,
        "times": times[indices],
        # ForwardResult retains the PUNQ-oriented historical field name. For
        # Egg this value is the terminal (10-year) FOPT from the supplied deck.
        "FOPT_16.5y": float(vectors["FOPT"][-1]),
    }
