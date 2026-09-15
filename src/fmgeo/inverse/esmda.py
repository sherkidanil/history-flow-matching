"""Ensemble smoother with multiple data assimilation."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def validate_inflations(inflations: Sequence[float]) -> tuple[float, ...]:
    """Validate the ES-MDA condition that reciprocal inflations sum to one."""
    values = tuple(float(value) for value in inflations)
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("inflations must be finite and positive")
    if not math.isclose(sum(1.0 / value for value in values), 1.0, abs_tol=1e-8):
        raise ValueError("reciprocal inflation factors must sum to one")
    return values


def _truncated_inverse(matrix: NDArray[np.float64], energy: float) -> NDArray[np.float64]:
    if not 0.0 < energy <= 1.0:
        raise ValueError("svd_energy must be in (0, 1]")
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    if singular_values[0] <= 0:
        raise ValueError("innovation covariance must be positive definite")
    cumulative = np.cumsum(singular_values) / singular_values.sum()
    rank = min(int(np.searchsorted(cumulative, energy, side="left")) + 1, len(singular_values))
    cutoff = np.finfo(np.float64).eps * max(matrix.shape) * singular_values[0]
    rank = min(rank, int(np.count_nonzero(singular_values > cutoff)))
    if rank == 0:
        raise ValueError("innovation covariance is numerically singular")
    return (right[:rank].T / singular_values[:rank]) @ left[:, :rank].T


def esmda_update(
    parameters: ArrayLike,
    simulated_data: ArrayLike,
    *,
    observation: ArrayLike,
    observation_covariance: ArrayLike,
    inflation: float,
    rng: np.random.Generator,
    localization: ArrayLike | None = None,
    svd_energy: float = 0.999,
) -> NDArray[np.float64]:
    """Perform one stochastic ES-MDA update on ensemble-first arrays."""
    model = np.asarray(parameters, dtype=np.float64)
    data = np.asarray(simulated_data, dtype=np.float64)
    observed = np.asarray(observation, dtype=np.float64)
    covariance = np.asarray(observation_covariance, dtype=np.float64)
    if model.ndim != 2 or data.ndim != 2 or model.shape[0] != data.shape[0]:
        raise ValueError("parameters and simulated_data must be ensemble-first matrices")
    if model.shape[0] < 2:
        raise ValueError("at least two ensemble members are required")
    if observed.shape != (data.shape[1],) or covariance.shape != (data.shape[1], data.shape[1]):
        raise ValueError("observation dimensions do not match simulated data")
    if inflation <= 0 or not math.isfinite(inflation):
        raise ValueError("inflation must be finite and positive")
    if not np.allclose(covariance, covariance.T):
        raise ValueError("observation covariance must be symmetric")

    model_anomalies = model - model.mean(axis=0)
    data_anomalies = data - data.mean(axis=0)
    denominator = model.shape[0] - 1
    cross_covariance = model_anomalies.T @ data_anomalies / denominator
    if localization is not None:
        taper = np.asarray(localization, dtype=np.float64)
        if taper.shape != cross_covariance.shape:
            raise ValueError("localization shape must match parameter-data covariance")
        cross_covariance *= taper
    data_covariance = data_anomalies.T @ data_anomalies / denominator
    gain = cross_covariance @ _truncated_inverse(
        data_covariance + inflation * covariance, svd_energy
    )
    perturbed = rng.multivariate_normal(observed, inflation * covariance, size=model.shape[0])
    return np.asarray(model + (perturbed - data) @ gain.T, dtype=np.float64)

