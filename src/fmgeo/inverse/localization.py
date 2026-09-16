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


def well_observation_localization(
    active: ArrayLike,
    *,
    producer_locations_yx: tuple[tuple[int, int], ...],
    observation_times: int,
    cell_size_yx_m: tuple[float, float],
    radius_m: float,
    rate_types: int = 2,
) -> NDArray[np.float64]:
    """Build active-cell to well-observation Gaspari-Cohn tapers.

    ``radius_m`` is the Gaspari-Cohn half-width; taper support ends at twice
    this distance. Observations are ordered by rate type, producer, then time.
    """

    mask = np.asarray(active, dtype=bool)
    if mask.ndim != 3 or not np.any(mask):
        raise ValueError("active must be a non-empty 3D mask")
    if not producer_locations_yx or observation_times < 1 or rate_types < 1:
        raise ValueError("producer locations, observation times, and rate types are required")
    dy, dx = (float(value) for value in cell_size_yx_m)
    if dy <= 0 or dx <= 0:
        raise ValueError("cell sizes must be positive")
    coordinates = np.argwhere(mask)
    cell_y = coordinates[:, 1, None]
    cell_x = coordinates[:, 2, None]
    wells = np.asarray(producer_locations_yx, dtype=np.float64)
    if wells.ndim != 2 or wells.shape[1] != 2:
        raise ValueError("producer locations must contain (y, x) pairs")
    y_distance = (cell_y - wells[None, :, 0]) * dy
    x_distance = (cell_x - wells[None, :, 1]) * dx
    distances = np.sqrt(y_distance**2 + x_distance**2)
    per_well = gaspari_cohn(distances, half_width=radius_m)
    one_rate = np.repeat(per_well, observation_times, axis=1)
    return np.tile(one_rate, (1, rate_types))
