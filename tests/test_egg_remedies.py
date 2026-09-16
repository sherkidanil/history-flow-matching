from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from fmgeo.forward.runner import ForwardResult
from fmgeo.inverse.diagnostics import ensemble_collapse_diagnostics
from fmgeo.inverse.egg import load_egg_inversion_config, run_egg_esmda
from fmgeo.inverse.localization import well_observation_localization
from fmgeo.param.pca import PCAParameterization

repository = Path(__file__).parents[1]
sys.path.insert(0, str(repository / "scripts"))
from m8_invert_egg import _remedy_name, _validate_remedy  # noqa: E402

sys.path.pop(0)


def _scalar_forward(field: np.ndarray) -> ForwardResult:
    value = float(field.ravel()[0])
    return ForwardResult("ok", (value,), value, 0.0, 0, "")


def test_egg_esmda_propagates_localization_to_parameter_update() -> None:
    initial = np.column_stack((np.linspace(-1.0, 1.0, 100),) * 2)

    stages = run_egg_esmda(
        initial,
        decode=lambda values: values[:, :1, None, None],
        evaluator=_scalar_forward,
        observation=np.array([1.0]),
        observation_covariance=np.array([[0.01]]),
        inflations=(1.0,),
        rng=np.random.default_rng(7),
        workers=1,
        svd_energy=1.0,
        localization=np.array([[1.0], [0.0]]),
    )

    assert stages[-1].parameters[:, 0].mean() > 0.9
    np.testing.assert_array_equal(stages[-1].parameters[:, 1], initial[:, 1])


def test_well_localization_matches_active_cells_and_observation_order() -> None:
    active = np.ones((1, 2, 3), dtype=bool)
    taper = well_observation_localization(
        active,
        producer_locations_yx=((0, 0), (1, 2)),
        observation_times=2,
        cell_size_yx_m=(10.0, 10.0),
        radius_m=10.0,
        rate_types=2,
    )

    assert taper.shape == (6, 8)
    np.testing.assert_array_equal(taper[:, 0], taper[:, 1])
    np.testing.assert_array_equal(taper[:, 0], taper[:, 4])
    assert taper[0, 0] == pytest.approx(1.0)
    assert taper[-1, 2] == pytest.approx(1.0)
    assert taper[-1, 0] == pytest.approx(0.0)


def test_pca_can_match_an_explicit_latent_rank() -> None:
    ensemble = np.diag(np.arange(1.0, 7.0))

    model = PCAParameterization.fit(ensemble, variance_fraction=1.0, max_rank=3)
    scores = model.encode(ensemble)

    assert model.rank == 3
    assert scores.shape == (6, 3)
    np.testing.assert_allclose(model.encode(model.decode(scores)), scores, atol=1e-12)


def test_collapse_diagnostics_measure_spread_shift_and_effective_members() -> None:
    prior = np.array([[[[0.0, 0.0]]], [[[2.0, 0.0]]], [[[0.0, 2.0]]]])
    posterior = prior + 1.0

    diagnostics = ensemble_collapse_diagnostics(
        posterior,
        prior_fields=prior,
        active=np.ones((1, 1, 2), dtype=bool),
    )

    assert diagnostics["mean_field_spread"] == pytest.approx(1.1547005383792517)
    assert diagnostics["relative_field_shift_from_prior"] == pytest.approx(
        np.linalg.norm(posterior - prior) / np.linalg.norm(prior)
    )
    assert diagnostics["effective_ensemble_members"] == pytest.approx(2.6)


def test_collapse_diagnostics_rejects_mismatched_prior() -> None:
    with pytest.raises(ValueError, match="same shape"):
        ensemble_collapse_diagnostics(
            np.zeros((3, 1, 1, 2)),
            prior_fields=np.zeros((2, 1, 1, 2)),
            active=np.ones((1, 1, 2), dtype=bool),
        )


def test_remedy_metadata_and_pca_localization_guard() -> None:
    config = load_egg_inversion_config(repository / "configs/egg/inversion.yaml")
    localized = config.model_copy(
        update={"localization": config.localization.model_copy(update={"enabled": True})}
    )

    assert (
        _remedy_name(
            assimilation_count=8,
            localization_enabled=False,
            fm_latent_pca=False,
        )
        == "assimilation_steps"
    )
    with pytest.raises(ValueError, match="not applicable"):
        _validate_remedy("pca", localized)
