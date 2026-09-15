from __future__ import annotations

import numpy as np

from fmgeo.metrics.egg import egg_distribution_metrics


def test_identical_egg_distributions_have_zero_comparison_errors() -> None:
    rng = np.random.default_rng(17)
    fields = rng.normal(size=(4, 3, 8, 10))
    fields[:, :, 3:5, :] += 2.0
    active = np.ones((3, 8, 10), dtype=bool)

    metrics = egg_distribution_metrics(
        fields,
        fields.copy(),
        active_mask=active,
        high_permeability_threshold=1.0,
        seed=9,
        max_sample_cells=10_000,
        max_sample_fields=10,
        max_variogram_lag=3,
    )

    assert metrics["schema_version"] == 1
    assert metrics["marginal_ks"] == 0.0
    assert metrics["variogram_x_nrmse"] == 0.0
    assert metrics["variogram_y_nrmse"] == 0.0
    assert metrics["spanning_fraction_abs_error"] == 0.0
    assert metrics["bimodality_abs_error"] == 0.0


def test_spanning_metric_uses_active_x_boundaries() -> None:
    active = np.ones((1, 5, 7), dtype=bool)
    connected = np.zeros((1, 1, 5, 7), dtype=float)
    connected[0, 0, 2, :] = 2.0
    disconnected = connected.copy()
    disconnected[0, 0, 2, 3] = 0.0

    connected_metrics = egg_distribution_metrics(
        connected,
        connected,
        active_mask=active,
        high_permeability_threshold=1.0,
        seed=1,
        max_variogram_lag=2,
    )
    disconnected_metrics = egg_distribution_metrics(
        disconnected,
        connected,
        active_mask=active,
        high_permeability_threshold=1.0,
        seed=1,
        max_variogram_lag=2,
    )

    assert connected_metrics["spanning_fraction_generated"] == 1.0
    assert disconnected_metrics["spanning_fraction_generated"] == 0.0
