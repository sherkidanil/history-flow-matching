"""Deterministic spatial-resolution transforms with physical-scale bookkeeping."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray


def average_pool_horizontal(
    fields: NDArray[np.floating],
    *,
    active_mask: NDArray[np.bool_] | None = None,
    factor: int = 2,
) -> tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """Average the last two axes, excluding inactive cells from block means."""
    values = np.asarray(fields)
    if values.ndim < 3 or not np.issubdtype(values.dtype, np.floating):
        raise ValueError("fields must be a floating array ending in (z, y, x)")
    if factor < 1 or values.shape[-2] % factor or values.shape[-1] % factor:
        raise ValueError("horizontal dimensions must be divisible by a positive factor")
    if not np.all(np.isfinite(values)):
        raise ValueError("fields must be finite")
    spatial_shape = values.shape[-3:]
    active = (
        np.ones(spatial_shape, dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, dtype=bool)
    )
    if active.shape != spatial_shape or not np.any(active):
        raise ValueError("active_mask must match the non-empty spatial shape")

    coarse_y = spatial_shape[-2] // factor
    coarse_x = spatial_shape[-1] // factor
    blocked_shape = (*values.shape[:-2], coarse_y, factor, coarse_x, factor)
    blocked = values.reshape(blocked_shape)
    mask = active.reshape(spatial_shape[0], coarse_y, factor, coarse_x, factor)
    broadcast_mask = mask.reshape((1,) * (values.ndim - 3) + mask.shape)
    numerator = np.where(broadcast_mask, blocked, 0).sum(axis=(-3, -1), dtype=values.dtype)
    counts = mask.sum(axis=(-3, -1))
    pooled = np.divide(
        numerator,
        counts,
        out=np.zeros_like(numerator),
        where=counts > 0,
    )
    return pooled, counts > 0


def scale_correlation_lengths(
    corr_len_cells: Sequence[float],
    *,
    reference_shape: Sequence[int],
    target_shape: Sequence[int],
) -> tuple[float, float, float]:
    """Convert cell-based lengths while preserving physical domain lengths."""
    lengths = tuple(float(value) for value in corr_len_cells)
    reference = tuple(int(value) for value in reference_shape)
    target = tuple(int(value) for value in target_shape)
    if len(lengths) != 3 or len(reference) != 3 or len(target) != 3:
        raise ValueError("correlation lengths and shapes must contain three values")
    if any(value <= 0 for value in lengths) or any(value <= 0 for value in (*reference, *target)):
        raise ValueError("correlation lengths and shapes must be positive")
    scaled = tuple(
        length * target_size / reference_size
        for length, reference_size, target_size in zip(
            lengths, reference, target, strict=True
        )
    )
    return scaled[0], scaled[1], scaled[2]
