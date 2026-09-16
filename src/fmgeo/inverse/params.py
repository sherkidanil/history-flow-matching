"""Gaussian latent transforms for bounded physical parameters."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr, ndtri


def _validate_bounds(lower: float, upper: float) -> None:
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("lower bound must be finite and smaller than upper bound")


def bounded_from_normal(
    latent: ArrayLike, *, lower: float, upper: float
) -> NDArray[np.float64]:
    """Map standard-normal variables into an open bounded interval."""
    _validate_bounds(lower, upper)
    values = np.asarray(latent, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("latent values must be finite")
    return lower + (upper - lower) * ndtr(values)


def normal_from_bounded(
    physical: ArrayLike, *, lower: float, upper: float
) -> NDArray[np.float64]:
    """Invert :func:`bounded_from_normal` for strictly interior values."""
    _validate_bounds(lower, upper)
    values = np.asarray(physical, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("physical values must be finite")
    if np.any(values <= lower) or np.any(values >= upper):
        raise ValueError("physical values must lie strictly inside the bounds")
    probabilities = (values - lower) / (upper - lower)
    return np.asarray(ndtri(probabilities), dtype=np.float64)

