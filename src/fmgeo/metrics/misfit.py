"""Observation-space history-matching diagnostics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def normalized_data_misfit(
    observation: ArrayLike,
    simulated: ArrayLike,
    covariance: ArrayLike,
) -> float:
    """Evaluate the covariance-weighted quadratic residual per observation."""
    observed = np.asarray(observation, dtype=np.float64)
    predicted = np.asarray(simulated, dtype=np.float64)
    errors = np.asarray(covariance, dtype=np.float64)
    if observed.ndim != 1 or predicted.shape != observed.shape:
        raise ValueError("observation and simulated vectors must have matching shapes")
    if errors.shape != (observed.size, observed.size):
        raise ValueError("covariance shape must match the observation vector")
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(predicted)):
        raise ValueError("observation and simulated values must be finite")
    if not np.allclose(errors, errors.T):
        raise ValueError("covariance must be symmetric")
    np.linalg.cholesky(errors)
    residual = observed - predicted
    return float(residual @ np.linalg.solve(errors, residual) / observed.size)

