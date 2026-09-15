from __future__ import annotations

import numpy as np
import pytest

from fmgeo.inverse.params import bounded_from_normal, normal_from_bounded


def test_probit_transform_round_trip_and_bounds() -> None:
    latent = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])

    physical = bounded_from_normal(latent, lower=0.15, upper=0.25)
    recovered = normal_from_bounded(physical, lower=0.15, upper=0.25)

    assert np.all((physical > 0.15) & (physical < 0.25))
    np.testing.assert_allclose(recovered, latent, rtol=0, atol=1e-10)


def test_probit_rejects_invalid_bounds_and_boundary_values() -> None:
    with pytest.raises(ValueError, match="lower"):
        bounded_from_normal(np.array([0.0]), lower=1.0, upper=1.0)
    with pytest.raises(ValueError, match="strictly inside"):
        normal_from_bounded(np.array([0.15]), lower=0.15, upper=0.25)

