from __future__ import annotations

import numpy as np
import pytest

from fmgeo.metrics.misfit import normalized_data_misfit
from fmgeo.metrics.uq import coverage, crps_ensemble, forecast_quantiles, interval_width


def test_forecast_quantiles_coverage_and_width() -> None:
    values = np.arange(11, dtype=float)

    quantiles = forecast_quantiles(values)

    assert quantiles == {"P10": 1.0, "P50": 5.0, "P90": 9.0}
    assert coverage(5.0, lower=quantiles["P10"], upper=quantiles["P90"])
    assert not coverage(10.0, lower=quantiles["P10"], upper=quantiles["P90"])
    assert interval_width(quantiles["P10"], quantiles["P90"]) == 8.0


def test_crps_matches_direct_ensemble_definition() -> None:
    ensemble = np.array([0.0, 2.0])

    assert crps_ensemble(ensemble, observation=1.0) == pytest.approx(0.5)


def test_normalized_misfit_matches_quadratic_form() -> None:
    observation = np.array([2.0, 4.0])
    simulated = np.array([1.0, 2.0])
    covariance = np.diag([1.0, 4.0])

    assert normalized_data_misfit(observation, simulated, covariance) == pytest.approx(1.0)


def test_metrics_reject_nonfinite_values_and_reversed_interval() -> None:
    with pytest.raises(ValueError, match="finite"):
        forecast_quantiles(np.array([1.0, np.nan]))
    with pytest.raises(ValueError, match="lower"):
        interval_width(2.0, 1.0)

