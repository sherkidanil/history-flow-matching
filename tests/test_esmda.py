from __future__ import annotations

import numpy as np
import pytest

from fmgeo.inverse.esmda import esmda_update, validate_inflations
from fmgeo.inverse.localization import gaspari_cohn


def test_inflation_schedule_requires_reciprocal_sum_one() -> None:
    assert validate_inflations([4.0, 4.0, 4.0, 4.0]) == (4.0, 4.0, 4.0, 4.0)
    with pytest.raises(ValueError, match="sum"):
        validate_inflations([2.0, 4.0])


def test_esmda_matches_scalar_linear_gaussian_posterior() -> None:
    rng = np.random.default_rng(812)
    prior = rng.normal(0.0, 1.0, size=(30_000, 1))
    simulated = prior.copy()

    posterior = esmda_update(
        prior,
        simulated,
        observation=np.array([1.0]),
        observation_covariance=np.array([[1.0]]),
        inflation=1.0,
        rng=np.random.default_rng(99),
    )

    assert posterior[:, 0].mean() == pytest.approx(0.5, abs=0.02)
    assert posterior[:, 0].var(ddof=1) == pytest.approx(0.5, abs=0.02)


def test_esmda_is_reproducible_with_equal_generators() -> None:
    parameters = np.arange(20, dtype=float).reshape(10, 2)
    simulated = parameters[:, :1]
    kwargs = {
        "observation": np.array([3.0]),
        "observation_covariance": np.array([[0.5]]),
        "inflation": 2.0,
    }

    left = esmda_update(parameters, simulated, rng=np.random.default_rng(4), **kwargs)
    right = esmda_update(parameters, simulated, rng=np.random.default_rng(4), **kwargs)

    np.testing.assert_array_equal(left, right)


def test_gaspari_cohn_support_and_endpoints() -> None:
    distances = np.array([0.0, 1.0, 2.0, 2.1])
    taper = gaspari_cohn(distances, half_width=1.0)

    assert taper[0] == pytest.approx(1.0)
    assert taper[2] == pytest.approx(0.0, abs=1e-12)
    assert taper[3] == 0.0
    assert np.all((taper >= 0.0) & (taper <= 1.0))

