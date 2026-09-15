from __future__ import annotations

import numpy as np

from fmgeo.metrics.egg import (
    assess_egg_generator,
    bimodality_coefficient,
    egg_distribution_metrics,
    egg_well_connectivity,
)


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


def test_egg_well_connectivity_returns_all_injector_producer_pairs() -> None:
    connected = np.zeros((1, 2, 3, 4), dtype=float)
    connected[0, 0, 1, :] = 2.0
    wells = {"INJECT1": (1, 0), "PROD1": (1, 3), "PROD2": (2, 3)}

    result = egg_well_connectivity(connected, threshold=1.0, wells=wells)

    assert result == {("INJECT1", "PROD1"): 1.0, ("INJECT1", "PROD2"): 0.0}


def test_generator_acceptance_requires_every_shared_limit() -> None:
    metrics = {
        "marginal_ks": 0.05,
        "variogram_x_nrmse": 0.20,
        "variogram_y_nrmse": 0.30,
        "spanning_fraction_abs_error": 0.04,
        "bimodality_abs_error": 0.01,
    }
    limits = {
        "roundtrip_relative_error": 0.01,
        "marginal_ks": 0.10,
        "variogram_x_nrmse": 0.25,
        "variogram_y_nrmse": 0.25,
        "spanning_fraction_abs_error": 0.10,
        "bimodality_abs_error": 0.05,
    }

    result = assess_egg_generator(metrics, roundtrip_relative_error=0.005, limits=limits)

    assert not result["accepted"]
    assert not result["criteria"]["variogram_y_nrmse"]["passed"]
    assert result["criteria"]["marginal_ks"]["passed"]


def test_bimodality_coefficient_is_high_for_two_separated_modes() -> None:
    values = np.repeat([-1.0, 1.0], 100)

    assert bimodality_coefficient(values) > 0.95
