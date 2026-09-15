"""Deterministic, geometry-aware augmentation of Egg ensembles."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from numpy.typing import NDArray


class CropMetadata(TypedDict):
    """Provenance and transforms applied to one augmented crop."""

    source_index: int
    z_start: int
    y_start: int
    x_start: int
    flip_z: bool
    flip_y: bool
    flip_x: bool
    rotation_quarters: int


def directional_rotation_allowed(
    ensemble: NDArray[np.generic], *, relative_tolerance: float
) -> bool:
    """Return whether swapping horizontal axes preserves lag-one variability.

    A quarter-turn rotation is accepted only when the ensemble's mean squared
    lag-one increments in x and y agree within ``relative_tolerance``.  This is
    a deliberately conservative directional-variogram gate.
    """

    values = np.asarray(ensemble, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("ensemble must have shape (realization, z, y, x)")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")
    if values.shape[-2] < 2 or values.shape[-1] < 2:
        return False

    gamma_x = float(np.mean(np.diff(values, axis=-1) ** 2) / 2.0)
    gamma_y = float(np.mean(np.diff(values, axis=-2) ** 2) / 2.0)
    scale = max(gamma_x, gamma_y)
    if scale == 0.0:
        return True
    return abs(gamma_x - gamma_y) / scale <= relative_tolerance


def augment_crops(
    ensemble: NDArray[np.generic],
    *,
    crop_shape: tuple[int, int, int],
    count: int,
    seed: int,
    rotations: bool,
    rotation_variogram_tolerance: float = 0.2,
    flip_axes_zyx: tuple[bool, bool, bool] = (False, True, True),
) -> tuple[NDArray[np.generic], tuple[CropMetadata, ...]]:
    """Sample deterministic crops and apply reflections and safe rotations."""

    values = np.asarray(ensemble)
    if values.ndim != 4:
        raise ValueError("ensemble must have shape (realization, z, y, x)")
    if count < 1:
        raise ValueError("count must be positive")
    if len(crop_shape) != 3 or any(size < 1 for size in crop_shape):
        raise ValueError("crop_shape must contain three positive sizes")

    spatial_shape = values.shape[1:]
    if any(crop > size for crop, size in zip(crop_shape, spatial_shape, strict=True)):
        raise ValueError("crop_shape must fit inside the ensemble spatial shape")
    if rotations and crop_shape[-2] != crop_shape[-1]:
        raise ValueError("horizontal crop dimensions must match when rotations are enabled")

    allow_rotations = rotations and directional_rotation_allowed(
        values, relative_tolerance=rotation_variogram_tolerance
    )
    rng = np.random.default_rng(seed)
    result = np.empty((count, *crop_shape), dtype=values.dtype)
    metadata: list[CropMetadata] = []

    for index in range(count):
        source_index = int(rng.integers(values.shape[0]))
        starts = tuple(
            int(rng.integers(size - crop + 1))
            for size, crop in zip(spatial_shape, crop_shape, strict=True)
        )
        z_start, y_start, x_start = starts
        z_size, y_size, x_size = crop_shape
        crop_values = values[
            source_index,
            z_start : z_start + z_size,
            y_start : y_start + y_size,
            x_start : x_start + x_size,
        ].copy()

        flips = tuple(
            enabled and bool(value)
            for enabled, value in zip(
                flip_axes_zyx, rng.integers(0, 2, size=3), strict=True
            )
        )
        for axis, flip in enumerate(flips):
            if flip:
                crop_values = np.flip(crop_values, axis=axis)

        rotation_quarters = int(rng.integers(0, 4)) if allow_rotations else 0
        if rotation_quarters:
            crop_values = np.rot90(crop_values, rotation_quarters, axes=(-2, -1))

        result[index] = crop_values
        metadata.append(
            CropMetadata(
                source_index=source_index,
                z_start=z_start,
                y_start=y_start,
                x_start=x_start,
                flip_z=flips[0],
                flip_y=flips[1],
                flip_x=flips[2],
                rotation_quarters=rotation_quarters,
            )
        )

    return result, tuple(metadata)
