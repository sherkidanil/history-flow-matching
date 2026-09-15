"""One predeclared metric schema for all Egg prior and FM comparisons."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import ndimage, stats

from fmgeo.metrics.geology import experimental_variogram

EggMetricValue = float | int


def _sample_active_values(
    fields: NDArray[np.float64],
    active: NDArray[np.bool_],
    *,
    count: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    cells = np.flatnonzero(active.ravel())
    total = fields.shape[0] * len(cells)
    if total <= count:
        return np.asarray(fields[:, active].ravel(), dtype=np.float64)
    realization_indices = rng.integers(fields.shape[0], size=count)
    cell_indices = cells[rng.integers(len(cells), size=count)]
    flattened = fields.reshape(fields.shape[0], -1)
    return np.asarray(flattened[realization_indices, cell_indices], dtype=np.float64)


def _normalized_rmse(values: NDArray[np.float64], reference: NDArray[np.float64]) -> float:
    difference = values[1:] - reference[1:]
    scale = float(np.sqrt(np.mean(reference[1:] ** 2)))
    error = float(np.sqrt(np.mean(difference**2)))
    if scale == 0.0:
        return 0.0 if error == 0.0 else float("inf")
    return error / scale


def _spans_x(high: NDArray[np.bool_], active: NDArray[np.bool_]) -> bool:
    active_x = np.flatnonzero(np.any(active, axis=(0, 1)))
    labels, _ = ndimage.label(high & active, structure=ndimage.generate_binary_structure(3, 1))
    left_mask = high[..., active_x[0]] & active[..., active_x[0]]
    left = set(np.unique(labels[..., active_x[0]][left_mask]))
    right = set(
        np.unique(labels[..., active_x[-1]][high[..., active_x[-1]] & active[..., active_x[-1]]])
    )
    return bool((left & right) - {0})


def _bimodality_coefficient(values: NDArray[np.float64]) -> float:
    kurtosis = float(stats.kurtosis(values, fisher=False, bias=False))
    if not np.isfinite(kurtosis) or kurtosis <= 0:
        return float("nan")
    skewness = float(stats.skew(values, bias=False))
    return (skewness**2 + 1.0) / kurtosis


def egg_distribution_metrics(
    generated: ArrayLike,
    reference: ArrayLike,
    *,
    active_mask: ArrayLike,
    high_permeability_threshold: float,
    seed: int,
    max_sample_cells: int = 200_000,
    max_sample_fields: int = 128,
    max_variogram_lag: int = 10,
) -> dict[str, EggMetricValue]:
    """Compare two Egg ensembles without loading all cell pairs into memory."""

    samples = np.asarray(generated, dtype=np.float64)
    truth = np.asarray(reference, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    if samples.ndim != 4 or truth.ndim != 4 or samples.shape[1:] != truth.shape[1:]:
        raise ValueError("generated and reference must share spatial shape (z, y, x)")
    if active.shape != samples.shape[1:] or not np.any(active):
        raise ValueError("active_mask must match the non-empty spatial grid")
    if samples.shape[0] < 1 or truth.shape[0] < 1:
        raise ValueError("both ensembles must contain at least one field")
    if max_sample_cells < 2 or max_sample_fields < 1:
        raise ValueError("metric sampling limits must be positive")
    if max_variogram_lag < 1 or max_variogram_lag >= min(active.shape[-2:]):
        raise ValueError("max_variogram_lag must fit both horizontal axes")

    rng = np.random.default_rng(seed)
    sampled_values = _sample_active_values(samples, active, count=max_sample_cells, rng=rng)
    reference_values = _sample_active_values(truth, active, count=max_sample_cells, rng=rng)
    generated_subset = samples[:max_sample_fields]
    reference_subset = truth[:max_sample_fields]
    variogram_x = experimental_variogram(
        generated_subset, axis=2, max_lag=max_variogram_lag, active_mask=active
    )
    reference_variogram_x = experimental_variogram(
        reference_subset, axis=2, max_lag=max_variogram_lag, active_mask=active
    )
    variogram_y = experimental_variogram(
        generated_subset, axis=1, max_lag=max_variogram_lag, active_mask=active
    )
    reference_variogram_y = experimental_variogram(
        reference_subset, axis=1, max_lag=max_variogram_lag, active_mask=active
    )
    spanning_generated = float(
        np.mean(
            [_spans_x(field >= high_permeability_threshold, active) for field in generated_subset]
        )
    )
    spanning_reference = float(
        np.mean(
            [_spans_x(field >= high_permeability_threshold, active) for field in reference_subset]
        )
    )
    generated_bimodality = _bimodality_coefficient(sampled_values)
    reference_bimodality = _bimodality_coefficient(reference_values)
    return {
        "schema_version": 1,
        "metric_sample_cells_generated": len(sampled_values),
        "metric_sample_cells_reference": len(reference_values),
        "metric_sample_fields_generated": len(generated_subset),
        "metric_sample_fields_reference": len(reference_subset),
        "marginal_ks": float(stats.ks_2samp(sampled_values, reference_values).statistic),
        "variogram_x_nrmse": _normalized_rmse(variogram_x, reference_variogram_x),
        "variogram_y_nrmse": _normalized_rmse(variogram_y, reference_variogram_y),
        "spanning_fraction_generated": spanning_generated,
        "spanning_fraction_reference": spanning_reference,
        "spanning_fraction_abs_error": abs(spanning_generated - spanning_reference),
        "bimodality_generated": generated_bimodality,
        "bimodality_reference": reference_bimodality,
        "bimodality_abs_error": abs(generated_bimodality - reference_bimodality),
    }
