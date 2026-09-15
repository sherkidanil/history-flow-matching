"""Metrics for geological structure and facies connectivity."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import generate_binary_structure, label


def facies_fraction_by_layer(
    labels: ArrayLike,
    *,
    facies: int,
    active_mask: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return the selected-facies fraction for each z layer."""
    values = np.asarray(labels)
    if values.ndim != 3:
        raise ValueError("labels must have shape (z, y, x)")
    active = (
        np.ones(values.shape, dtype=bool)
        if active_mask is None
        else np.asarray(active_mask, bool)
    )
    if active.shape != values.shape:
        raise ValueError("active_mask must match labels")
    counts = active.sum(axis=(1, 2))
    if np.any(counts == 0):
        raise ValueError("every layer must contain at least one active cell")
    return np.asarray(((values == facies) & active).sum(axis=(1, 2)) / counts, dtype=np.float64)


def experimental_variogram(
    field: ArrayLike,
    *,
    axis: int,
    max_lag: int,
    active_mask: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Compute a semivariogram along one spatial axis for a field or ensemble."""
    values = np.asarray(field, dtype=np.float64)
    if values.ndim not in (3, 4):
        raise ValueError("field must have shape (z,y,x) or (sample,z,y,x)")
    if axis not in (0, 1, 2):
        raise ValueError("axis must refer to z, y, or x")
    array_axis = axis + (1 if values.ndim == 4 else 0)
    if max_lag < 0 or max_lag >= values.shape[array_axis]:
        raise ValueError("max_lag must fit inside the requested axis")
    if not np.all(np.isfinite(values)):
        raise ValueError("field values must be finite")
    mask = None if active_mask is None else np.asarray(active_mask, dtype=bool)
    if mask is not None and mask.shape != values.shape[-3:]:
        raise ValueError("active_mask must match the spatial field shape")

    result = np.zeros(max_lag + 1, dtype=np.float64)
    for lag in range(1, max_lag + 1):
        left = [slice(None)] * values.ndim
        right = [slice(None)] * values.ndim
        left[array_axis] = slice(None, -lag)
        right[array_axis] = slice(lag, None)
        squared = (values[tuple(right)] - values[tuple(left)]) ** 2
        if mask is not None:
            mask_left = [slice(None)] * 3
            mask_right = [slice(None)] * 3
            mask_left[axis] = slice(None, -lag)
            mask_right[axis] = slice(lag, None)
            valid = mask[tuple(mask_left)] & mask[tuple(mask_right)]
            if values.ndim == 4:
                valid = np.broadcast_to(valid, squared.shape)
            if not np.any(valid):
                result[lag] = np.nan
                continue
            squared = squared[valid]
        result[lag] = 0.5 * squared.mean()
    return result


def _labeled_components(sand: NDArray[np.bool_]) -> tuple[NDArray[np.int32], int]:
    structure = generate_binary_structure(rank=3, connectivity=1)
    components, count = label(sand, structure=structure)
    return np.asarray(components, dtype=np.int32), int(count)


def connected_component_sizes(sand: ArrayLike) -> NDArray[np.int64]:
    """Return sorted 6-neighbor component sizes for a 3D sand mask."""
    values = np.asarray(sand, dtype=bool)
    if values.ndim != 3:
        raise ValueError("sand mask must have shape (z, y, x)")
    components, count = _labeled_components(values)
    if count == 0:
        return np.empty(0, dtype=np.int64)
    sizes = np.bincount(components.ravel())[1:]
    return np.sort(np.asarray(sizes, dtype=np.int64))


def well_pair_connected(
    sand: ArrayLike,
    first: Sequence[int],
    second: Sequence[int],
) -> bool:
    """Check whether two cells share a nonzero 6-neighbor sand component."""
    values = np.asarray(sand, dtype=bool)
    if values.ndim != 3 or len(first) != 3 or len(second) != 3:
        raise ValueError("sand and well coordinates must be three-dimensional")
    first_index = tuple(int(value) for value in first)
    second_index = tuple(int(value) for value in second)
    try:
        if not values[first_index] or not values[second_index]:
            return False
    except IndexError as error:
        raise ValueError("well coordinate is outside the grid") from error
    components, _ = _labeled_components(values)
    return bool(
        components[first_index] != 0
        and components[first_index] == components[second_index]
    )


def well_pair_connectivity_probability(
    ensemble: ArrayLike,
    first: Sequence[int],
    second: Sequence[int],
) -> float:
    """Estimate well-pair connection probability across facies realizations."""
    values = np.asarray(ensemble, dtype=bool)
    if values.ndim != 4 or values.shape[0] == 0:
        raise ValueError("ensemble must have shape (sample, z, y, x)")
    connections = [well_pair_connected(sample, first, second) for sample in values]
    return float(np.mean(connections))


def well_column_connected(
    sand: ArrayLike,
    first_yx: Sequence[int],
    second_yx: Sequence[int],
) -> bool:
    """Check whether any completed cells in two vertical wells share a component."""

    values = np.asarray(sand, dtype=bool)
    if values.ndim != 3 or len(first_yx) != 2 or len(second_yx) != 2:
        raise ValueError("sand must be 3D and well coordinates must be (y, x)")
    first = tuple(int(value) for value in first_yx)
    second = tuple(int(value) for value in second_yx)
    if any(
        coordinate < 0 or coordinate >= bound
        for pair in (first, second)
        for coordinate, bound in zip(pair, values.shape[1:], strict=True)
    ):
        raise ValueError("well coordinate is outside the grid")
    components, _ = _labeled_components(values)
    first_labels = set(np.unique(components[:, first[0], first[1]])) - {0}
    second_labels = set(np.unique(components[:, second[0], second[1]])) - {0}
    return bool(first_labels & second_labels)


def well_column_connectivity_probability(
    ensemble: ArrayLike,
    first_yx: Sequence[int],
    second_yx: Sequence[int],
) -> float:
    """Estimate vertical-well connection probability across an ensemble."""

    values = np.asarray(ensemble, dtype=bool)
    if values.ndim != 4 or values.shape[0] == 0:
        raise ValueError("ensemble must have shape (sample, z, y, x)")
    return float(
        np.mean(
            [well_column_connected(sample, first_yx, second_yx) for sample in values]
        )
    )
