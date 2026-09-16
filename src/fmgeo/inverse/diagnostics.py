"""Diagnostics for nonlinear ensemble parameterizations."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray


def sample_member_pairs(
    ensemble_size: int,
    pair_count: int,
    *,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Sample distinct unordered ensemble-member pairs without replacement."""

    if ensemble_size < 2 or pair_count < 1:
        raise ValueError("ensemble_size must exceed one and pair_count must be positive")
    first, second = np.triu_indices(ensemble_size, k=1)
    if pair_count > len(first):
        raise ValueError("pair_count exceeds the number of unique member pairs")
    selected = rng.choice(len(first), size=pair_count, replace=False)
    return np.column_stack((first[selected], second[selected])).astype(np.int64)


def midpoint_nonlinearity(
    parameters: ArrayLike,
    *,
    decode: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    pairs: ArrayLike,
    decoded_parameters: ArrayLike | None = None,
) -> NDArray[np.float64]:
    """Return relative midpoint-affinity errors for selected member pairs."""

    values = np.asarray(parameters, dtype=np.float64)
    indices = np.asarray(pairs, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] < 2 or not np.all(np.isfinite(values)):
        raise ValueError("parameters must be a finite ensemble-first matrix")
    if indices.ndim != 2 or indices.shape[1] != 2 or len(indices) == 0:
        raise ValueError("pairs must be a non-empty (pair, 2) integer matrix")
    if np.any(indices < 0) or np.any(indices >= len(values)):
        raise ValueError("pair index is outside the parameter ensemble")
    if np.any(indices[:, 0] == indices[:, 1]):
        raise ValueError("pair members must be distinct")

    decoded = (
        np.asarray(decode(values), dtype=np.float64)
        if decoded_parameters is None
        else np.asarray(decoded_parameters, dtype=np.float64)
    )
    if decoded.shape[0] != len(values) or not np.all(np.isfinite(decoded)):
        raise ValueError("decoded_parameters must be finite and ensemble-first")
    midpoints = 0.5 * (values[indices[:, 0]] + values[indices[:, 1]])
    decoded_midpoints = np.asarray(decode(midpoints), dtype=np.float64)
    expected = 0.5 * (decoded[indices[:, 0]] + decoded[indices[:, 1]])
    if decoded_midpoints.shape != expected.shape or not np.all(np.isfinite(decoded_midpoints)):
        raise ValueError("decoded midpoint shape or values are invalid")
    flattened_expected = expected.reshape(len(expected), -1)
    denominators = np.linalg.norm(flattened_expected, axis=1)
    if np.any(denominators <= 0):
        raise ValueError("decoded pair midpoints must have positive norm")
    differences = (decoded_midpoints - expected).reshape(len(expected), -1)
    return np.asarray(np.linalg.norm(differences, axis=1) / denominators, dtype=np.float64)


def relative_frobenius_shift(reference: ArrayLike, updated: ArrayLike) -> float:
    """Return the Frobenius update norm relative to the reference norm."""

    before = np.asarray(reference, dtype=np.float64)
    after = np.asarray(updated, dtype=np.float64)
    if before.shape != after.shape or before.size == 0:
        raise ValueError("reference and updated arrays must have the same non-empty shape")
    if not np.all(np.isfinite(before)) or not np.all(np.isfinite(after)):
        raise ValueError("reference and updated arrays must be finite")
    denominator = float(np.linalg.norm(before))
    if denominator <= 0:
        raise ValueError("reference array must have positive norm")
    return float(np.linalg.norm(after - before) / denominator)
