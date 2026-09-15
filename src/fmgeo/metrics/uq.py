"""Ensemble forecast uncertainty metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def _finite_ensemble(values: ArrayLike) -> np.ndarray:
    ensemble = np.asarray(values, dtype=np.float64)
    if ensemble.size == 0 or not np.all(np.isfinite(ensemble)):
        raise ValueError("ensemble must contain finite values")
    return ensemble


def forecast_quantiles(values: ArrayLike) -> dict[str, float]:
    """Return statistical 10th, 50th, and 90th percentiles."""
    ensemble = _finite_ensemble(values)
    p10, p50, p90 = np.quantile(ensemble, [0.1, 0.5, 0.9])
    return {"P10": float(p10), "P50": float(p50), "P90": float(p90)}


def coverage(truth: float, *, lower: float, upper: float) -> bool:
    """Return whether an inclusive interval covers a finite truth value."""
    if not np.all(np.isfinite([truth, lower, upper])):
        raise ValueError("truth and interval must be finite")
    if lower > upper:
        raise ValueError("lower interval endpoint exceeds upper endpoint")
    return lower <= truth <= upper


def interval_width(lower: float, upper: float) -> float:
    """Return a validated interval width."""
    if not np.all(np.isfinite([lower, upper])):
        raise ValueError("interval endpoints must be finite")
    if lower > upper:
        raise ValueError("lower interval endpoint exceeds upper endpoint")
    return float(upper - lower)


def crps_ensemble(values: ArrayLike, *, observation: float) -> float:
    """Compute empirical univariate CRPS from its energy representation."""
    ensemble = _finite_ensemble(values).reshape(-1)
    if not np.isfinite(observation):
        raise ValueError("observation must be finite")
    first = np.mean(np.abs(ensemble - observation))
    second = 0.5 * np.mean(np.abs(ensemble[:, None] - ensemble[None, :]))
    return float(first - second)


def energy_score(values: ArrayLike, *, observation: ArrayLike) -> float:
    """Compute the multivariate empirical energy score."""
    ensemble = _finite_ensemble(values)
    observed = np.asarray(observation, dtype=np.float64)
    if ensemble.ndim != 2 or observed.shape != (ensemble.shape[1],):
        raise ValueError("expected ensemble (sample, variable) and matching observation")
    if not np.all(np.isfinite(observed)):
        raise ValueError("observation must be finite")
    first = np.linalg.norm(ensemble - observed, axis=1).mean()
    pairwise = np.linalg.norm(ensemble[:, None, :] - ensemble[None, :, :], axis=2)
    return float(first - 0.5 * pairwise.mean())

