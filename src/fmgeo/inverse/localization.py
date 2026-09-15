"""Distance-dependent covariance localization."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def gaspari_cohn(distances: ArrayLike, *, half_width: float) -> NDArray[np.float64]:
    """Evaluate the fifth-order Gaspari-Cohn taper with support ``2*half_width``."""
    if not np.isfinite(half_width) or half_width <= 0:
        raise ValueError("half_width must be finite and positive")
    distance = np.asarray(distances, dtype=np.float64)
    if np.any(distance < 0) or not np.all(np.isfinite(distance)):
        raise ValueError("distances must be finite and non-negative")
    ratio = distance / half_width
    taper = np.zeros_like(ratio)
    inner = ratio <= 1.0
    r = ratio[inner]
    taper[inner] = 1.0 - 5.0 / 3.0 * r**2 + 5.0 / 8.0 * r**3 + 0.5 * r**4 - 0.25 * r**5
    outer = (ratio > 1.0) & (ratio < 2.0)
    r = ratio[outer]
    taper[outer] = (
        4.0
        - 5.0 * r
        + 5.0 / 3.0 * r**2
        + 5.0 / 8.0 * r**3
        - 0.5 * r**4
        + 1.0 / 12.0 * r**5
        - 2.0 / (3.0 * r)
    )
    return np.clip(taper, 0.0, 1.0)

