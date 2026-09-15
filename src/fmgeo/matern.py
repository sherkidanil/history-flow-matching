"""Anisotropic Matérn random fields sampled by padded FFT filtering."""

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial

import numpy as np
from numpy.typing import NDArray
from scipy.fft import next_fast_len
from scipy.optimize import minimize_scalar
from scipy.special import gamma, kv

FloatArray = NDArray[np.float64]


def matern_correlation(distance: float | FloatArray, nu: float = 1.5) -> FloatArray:
    """Evaluate unit-scale Matérn correlation using kappa=sqrt(2*nu)."""
    if nu <= 0:
        raise ValueError("nu must be positive")
    distances = np.asarray(distance, dtype=np.float64)
    if np.any(distances < 0):
        raise ValueError("distance must be non-negative")
    scaled = np.sqrt(2.0 * nu) * distances
    result = np.ones_like(scaled)
    positive = scaled > 0
    result[positive] = (
        2.0 ** (1.0 - nu)
        / gamma(nu)
        * scaled[positive] ** nu
        * kv(nu, scaled[positive])
    )
    return result


def matern_field(
    shape: Sequence[int],
    corr_len: Sequence[float],
    nu: float = 1.5,
    rng: np.random.Generator | None = None,
    pad: float = 2.0,
) -> FloatArray:
    """Sample a zero-mean, unit-variance 3D anisotropic Matérn field.

    Correlation lengths are expressed in grid cells in ``(z, y, x)`` order.
    Padding separates the returned crop from periodic FFT boundaries.
    """
    requested_shape = tuple(int(value) for value in shape)
    lengths = tuple(float(value) for value in corr_len)
    if len(requested_shape) != 3 or len(lengths) != 3:
        raise ValueError("shape and corr_len must contain exactly three values")
    if any(value <= 0 for value in requested_shape):
        raise ValueError("shape dimensions must be positive")
    if any(value <= 0 or not math.isfinite(value) for value in lengths):
        raise ValueError("correlation lengths must be finite and positive")
    if nu <= 0 or not math.isfinite(nu):
        raise ValueError("nu must be finite and positive")
    if pad < 1.5:
        raise ValueError("pad must be at least 1.5 to suppress periodic artifacts")

    generator = rng if rng is not None else np.random.default_rng()
    padded_shape = tuple(next_fast_len(math.ceil(pad * size)) for size in requested_shape)
    angular_frequencies = [
        2.0 * np.pi * np.fft.fftfreq(size) for size in padded_shape
    ]
    frequency_grids = np.meshgrid(*angular_frequencies, indexing="ij", sparse=True)
    scaled_frequency_squared = sum(
        (length * frequency) ** 2
        for length, frequency in zip(lengths, frequency_grids, strict=True)
    )
    kappa_squared = 2.0 * nu
    spectrum = (kappa_squared + scaled_frequency_squared) ** (-(nu + 1.5))
    spectrum /= spectrum.mean()

    white_noise = generator.standard_normal(padded_shape)
    filtered = np.fft.ifftn(np.fft.fftn(white_noise) * np.sqrt(spectrum)).real
    starts = tuple(
        (padded - size) // 2
        for padded, size in zip(padded_shape, requested_shape, strict=True)
    )
    crop = tuple(
        slice(start, start + size) for start, size in zip(starts, requested_shape, strict=True)
    )
    return np.asarray(filtered[crop], dtype=np.float64)


def empirical_axis_correlation(
    ensemble: NDArray[np.floating], axis: int, max_lag: int
) -> FloatArray:
    """Estimate stationary correlations along a spatial ensemble axis."""
    values = np.asarray(ensemble, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("ensemble must have shape (sample, z, y, x)")
    if axis not in (0, 1, 2):
        raise ValueError("axis must refer to z, y, or x")
    array_axis = axis + 1
    if max_lag < 0 or max_lag >= values.shape[array_axis]:
        raise ValueError("max_lag must fit inside the requested axis")
    centered = values - values.mean()
    variance = np.mean(centered**2)
    if variance == 0:
        raise ValueError("cannot correlate a constant ensemble")

    correlations = np.empty(max_lag + 1, dtype=np.float64)
    correlations[0] = 1.0
    for lag in range(1, max_lag + 1):
        left = [slice(None)] * values.ndim
        right = [slice(None)] * values.ndim
        left[array_axis] = slice(None, -lag)
        right[array_axis] = slice(lag, None)
        correlations[lag] = np.mean(centered[tuple(left)] * centered[tuple(right)]) / variance
    return correlations


def _fit_objective(
    length: float, *, lags: FloatArray, empirical: FloatArray, nu: float
) -> float:
    predicted = matern_correlation(lags / length, nu=nu)
    return float(np.mean((predicted - empirical[1:]) ** 2))


def fit_matern_to_ensemble(
    ensemble: NDArray[np.floating], nu: float = 1.5, max_lag: int = 8
) -> tuple[float, float, float]:
    """Fit independent axis correlation lengths to ensemble correlations."""
    values = np.asarray(ensemble, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("ensemble must have shape (sample, z, y, x)")
    fitted: list[float] = []
    for axis in range(3):
        axis_max_lag = min(max_lag, values.shape[axis + 1] - 1)
        empirical = empirical_axis_correlation(values, axis, axis_max_lag)
        lags = np.arange(1, axis_max_lag + 1, dtype=np.float64)

        result = minimize_scalar(
            partial(_fit_objective, lags=lags, empirical=empirical, nu=nu),
            bounds=(0.25, max(1.0, axis_max_lag * 5.0)),
            method="bounded",
            options={"xatol": 1e-4},
        )
        if not result.success:
            raise RuntimeError(f"Matérn fit failed on axis {axis}: {result.message}")
        fitted.append(float(result.x))
    return tuple(fitted)  # type: ignore[return-value]
