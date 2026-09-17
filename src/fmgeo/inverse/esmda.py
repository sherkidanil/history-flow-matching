"""Ensemble smoother with multiple data assimilation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class LinearResponseDiagnostic:
    """Variance explained by a truncated linear ensemble regression."""

    r2: float
    retained_rank: int
    available_rank: int


def validate_inflations(inflations: Sequence[float]) -> tuple[float, ...]:
    """Validate the ES-MDA condition that reciprocal inflations sum to one."""
    values = tuple(float(value) for value in inflations)
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("inflations must be finite and positive")
    if not math.isclose(sum(1.0 / value for value in values), 1.0, abs_tol=1e-8):
        raise ValueError("reciprocal inflation factors must sum to one")
    return values


def _energy_rank(weights: NDArray[np.float64], energy: float) -> int:
    if not 0.0 < energy <= 1.0:
        raise ValueError("svd_energy must be in (0, 1]")
    if weights.ndim != 1 or len(weights) == 0 or np.any(weights < 0):
        raise ValueError("SVD weights must be a non-empty non-negative vector")
    total = float(weights.sum())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("SVD weights must have positive finite mass")
    cumulative = np.cumsum(weights) / total
    return min(int(np.searchsorted(cumulative, energy, side="left")) + 1, len(weights))


def _truncated_inverse(matrix: NDArray[np.float64], energy: float) -> NDArray[np.float64]:
    left, singular_values, right = np.linalg.svd(matrix, full_matrices=False)
    if singular_values[0] <= 0:
        raise ValueError("innovation covariance must be positive definite")
    cutoff = np.finfo(np.float64).eps * max(matrix.shape) * singular_values[0]
    available_rank = int(np.count_nonzero(singular_values > cutoff))
    if available_rank == 0:
        raise ValueError("innovation covariance is numerically singular")
    rank = _energy_rank(singular_values[:available_rank], energy)
    return (right[:rank].T / singular_values[:rank]) @ left[:, :rank].T


def linear_response_diagnostic(
    parameters: ArrayLike,
    simulated_data: ArrayLike,
    *,
    svd_energy: float = 0.999,
) -> LinearResponseDiagnostic:
    """Measure data variance explained by truncated linear parameter regression."""

    model = np.asarray(parameters, dtype=np.float64)
    data = np.asarray(simulated_data, dtype=np.float64)
    if model.ndim != 2 or data.ndim != 2 or model.shape[0] != data.shape[0]:
        raise ValueError("parameters and simulated_data must be ensemble-first matrices")
    if model.shape[0] < 2:
        raise ValueError("at least two ensemble members are required")
    if not np.all(np.isfinite(model)) or not np.all(np.isfinite(data)):
        raise ValueError("parameters and simulated_data must be finite")
    model_anomalies = model - model.mean(axis=0)
    data_anomalies = data - data.mean(axis=0)
    left, singular_values, _ = np.linalg.svd(model_anomalies, full_matrices=False)
    if len(singular_values) == 0 or singular_values[0] <= 0:
        raise ValueError("parameter anomalies are numerically singular")
    cutoff = (
        np.finfo(np.float64).eps
        * max(model_anomalies.shape)
        * singular_values[0]
    )
    available_rank = int(np.count_nonzero(singular_values > cutoff))
    if available_rank == 0:
        raise ValueError("parameter anomalies are numerically singular")
    retained_rank = _energy_rank(singular_values[:available_rank] ** 2, svd_energy)
    response_energy = float(np.sum(data_anomalies**2))
    if not math.isfinite(response_energy) or response_energy <= 0:
        raise ValueError("simulated data anomalies must have positive finite variance")
    basis = left[:, :retained_rank]
    fitted = basis @ (basis.T @ data_anomalies)
    residual_energy = float(np.sum((data_anomalies - fitted) ** 2))
    r2 = float(np.clip(1.0 - residual_energy / response_energy, 0.0, 1.0))
    return LinearResponseDiagnostic(
        r2=r2,
        retained_rank=retained_rank,
        available_rank=available_rank,
    )


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
