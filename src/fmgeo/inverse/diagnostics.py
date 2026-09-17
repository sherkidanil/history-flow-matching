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


def ensemble_collapse_diagnostics(
    fields: ArrayLike,
    *,
    prior_fields: ArrayLike,
    active: ArrayLike,
) -> dict[str, float]:
    """Measure field spread, prior-relative shift, and anomaly participation.

    ``effective_ensemble_members`` is one plus the covariance-eigenvalue
    participation ratio. It ranges from one for an identical ensemble to the
    actual member count when all ``N - 1`` anomaly directions contribute equally.
    """

    values = np.asarray(fields, dtype=np.float64)
    prior = np.asarray(prior_fields, dtype=np.float64)
    mask = np.asarray(active, dtype=bool)
    if values.shape != prior.shape or values.ndim != 4:
        raise ValueError("fields and prior_fields must have the same shape")
    if mask.shape != values.shape[1:] or not np.any(mask):
        raise ValueError("active mask must match the field shape")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(prior)):
        raise ValueError("field ensembles must be finite")
    active_values = values[:, mask]
    anomalies = active_values - active_values.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(anomalies @ anomalies.T)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    energy = float(eigenvalues.sum())
    participation = (
        0.0 if energy == 0.0 else energy**2 / float(np.sum(eigenvalues**2))
    )
    effective = min(float(len(values)), 1.0 + participation)
    return {
        "mean_field_spread": float(np.std(active_values, axis=0, ddof=1).mean()),
        "relative_field_shift_from_prior": relative_frobenius_shift(
            prior[:, mask], active_values
        ),
        "effective_ensemble_members": effective,
    }
