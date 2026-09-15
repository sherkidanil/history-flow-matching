from __future__ import annotations

import numpy as np

from fmgeo.param.pca import PCAParameterization


def test_pca_round_trip_on_retained_subspace() -> None:
    coefficients = np.array([[-2.0, -1.0], [-1.0, 2.0], [1.0, -2.0], [2.0, 1.0]])
    basis = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 2.0, 0.0, -1.0]])
    fields = (coefficients @ basis).reshape(4, 2, 2)

    model = PCAParameterization.fit(fields, variance_fraction=1.0)
    reconstructed = model.decode(model.encode(fields))

    assert model.rank == 2
    np.testing.assert_allclose(reconstructed, fields, atol=1e-12)


def test_pca_uses_minimum_rank_reaching_variance_target() -> None:
    fields = np.array(
        [
            [-4.0, -1.0],
            [-2.0, 1.0],
            [2.0, -1.0],
            [4.0, 1.0],
        ]
    )

    model = PCAParameterization.fit(fields, variance_fraction=0.90)

    assert model.rank == 1
    assert model.explained_variance_fraction >= 0.90

