from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import ks_2samp

from fmgeo.matern import empirical_axis_correlation, fit_matern_to_ensemble, matern_field


def sample_ensemble(
    count: int,
    shape: tuple[int, int, int],
    corr_len: tuple[float, float, float],
    *,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([matern_field(shape, corr_len, nu=1.5, rng=rng) for _ in range(count)])


def test_matern_variance() -> None:
    fields = sample_ensemble(1000, (5, 9, 11), (1.2, 2.0, 2.8), seed=12)
    center_values = fields[:, 2, 4, 5]
    standard_error = np.sqrt(2.0 / (len(center_values) - 1))

    assert abs(center_values.mean()) < 3.0 / np.sqrt(len(center_values))
    assert abs(center_values.var(ddof=1) - 1.0) < 3.0 * standard_error


def test_matern_variogram_matches_theoretical_covariance() -> None:
    fields = sample_ensemble(700, (5, 11, 25), (1.2, 2.0, 3.0), seed=21)
    correlations = empirical_axis_correlation(fields, axis=2, max_lag=6)
    # For nu=1.5, rho(r)=(1+sqrt(3)r)exp(-sqrt(3)r).
    scaled = np.arange(7) / 3.0
    expected = (1.0 + np.sqrt(3.0) * scaled) * np.exp(-np.sqrt(3.0) * scaled)

    np.testing.assert_allclose(correlations[1:], expected[1:], rtol=0.05, atol=0.025)


def test_matern_anisotropy() -> None:
    requested = np.array([1.5, 3.0, 6.0])
    fields = sample_ensemble(350, (13, 21, 33), tuple(requested), seed=7)

    fitted = np.asarray(fit_matern_to_ensemble(fields, nu=1.5, max_lag=8))

    np.testing.assert_allclose(fitted, requested, rtol=0.10)


def test_matern_no_periodicity() -> None:
    fields = sample_ensemble(600, (5, 9, 25), (1.0, 1.5, 2.0), seed=91)
    correlations = empirical_axis_correlation(fields, axis=2, max_lag=24)

    assert abs(correlations[-1]) < 0.08
    assert correlations[-1] < correlations[1]


@pytest.mark.slow
def test_matern_resolution_consistency() -> None:
    rng_fine = np.random.default_rng(101)
    rng_coarse = np.random.default_rng(202)
    fine = np.stack(
        [matern_field((64, 64, 64), (12.0, 16.0, 20.0), rng=rng_fine) for _ in range(40)]
    )
    coarsened = fine.reshape(40, 32, 2, 32, 2, 32, 2).mean(axis=(2, 4, 6))
    coarse = np.stack(
        [matern_field((32, 32, 32), (6.0, 8.0, 10.0), rng=rng_coarse) for _ in range(40)]
    )
    coarsened = (coarsened - coarsened.mean()) / coarsened.std()
    coarse = (coarse - coarse.mean()) / coarse.std()

    assert ks_2samp(coarsened.ravel(), coarse.ravel()).statistic < 0.025
    for axis in range(3):
        left = empirical_axis_correlation(coarsened, axis=axis, max_lag=6)
        right = empirical_axis_correlation(coarse, axis=axis, max_lag=6)
        np.testing.assert_allclose(left[1:], right[1:], rtol=0.12, atol=0.04)
