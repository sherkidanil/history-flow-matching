from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from fmgeo.forward.runner import ForwardResult
from fmgeo.inverse.egg import (
    evaluate_egg_ensemble,
    load_egg_inversion_config,
    run_egg_esmda,
)

repository = Path(__file__).parents[1]
sys.path.insert(0, str(repository / "scripts"))
from egg_flow_adapter import embed_active  # noqa: E402
from m8_invert_egg import _resolve_parameterization_label  # noqa: E402

sys.path.pop(0)


def _write_config(path: Path, *, inflations: str = "[4.0, 4.0, 4.0, 4.0]") -> Path:
    path.write_text(
        f"""
seed: 19
truth_realization: 100
ensemble_size: 100
inflations: {inflations}
svd_energy: 0.999
pca_variance_fraction: 0.95
history_end_day: 1800.0
forecast_end_day: 3600.0
observation_interval_days: 90.0
oil_rate_relative_sigma: 0.05
water_rate_relative_sigma: 0.05
rate_sigma_floor: 1.0
water_breakthrough_fraction: 0.05
high_permeability_quantile: 0.65
workers: 16
timeout_seconds: 300.0
min_free_disk_gb: 20.0
""",
        encoding="utf-8",
    )
    return path


def test_egg_inversion_config_validates_standard_protocol(tmp_path: Path) -> None:
    config = load_egg_inversion_config(_write_config(tmp_path / "inversion.yaml"))

    assert config.truth_realization == 100
    assert config.ensemble_size == 100
    assert config.observation_count == 160
    assert config.inflations == (4.0, 4.0, 4.0, 4.0)


def test_parameterization_label_defaults_to_method_and_accepts_fm_variant() -> None:
    assert _resolve_parameterization_label("raw", None) == "raw"
    assert _resolve_parameterization_label("fm", "matern_misspec") == "matern_misspec"


def test_parameterization_label_rejects_empty_value() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        _resolve_parameterization_label("fm", "  ")


def test_embed_active_preserves_parameter_order() -> None:
    active = np.array([[[True, False], [False, True]]])
    parameters = np.array([[1.0, 2.0], [3.0, 4.0]])

    fields = embed_active(parameters, active)

    np.testing.assert_array_equal(fields[:, active], parameters)
    assert np.count_nonzero(fields[:, ~active]) == 0


def test_egg_inversion_config_rejects_invalid_inflations(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="sum"):
        load_egg_inversion_config(
            _write_config(tmp_path / "inversion.yaml", inflations="[2.0, 4.0]")
        )


def test_egg_ensemble_forward_preserves_member_order_and_failures() -> None:
    fields = np.arange(3 * 2 * 2 * 2, dtype=float).reshape(3, 2, 2, 2)

    def evaluate(field: np.ndarray) -> ForwardResult:
        member = int(field.ravel()[0] // 8)
        if member == 1:
            return ForwardResult("failed", None, None, 2.0, 1, "failed")
        return ForwardResult(
            "ok", (float(member), float(member + 1)), float(10 + member), 1.0, 0, ""
        )

    result = evaluate_egg_ensemble(fields, evaluator=evaluate, workers=2)

    np.testing.assert_array_equal(result.status, ["ok", "failed", "ok"])
    np.testing.assert_array_equal(result.simulated_data[[0, 2]], [[0.0, 1.0], [2.0, 3.0]])
    assert np.isnan(result.simulated_data[1]).all()
    np.testing.assert_array_equal(result.fopt[[0, 2]], [10.0, 12.0])
    assert np.isnan(result.fopt[1])


def test_egg_esmda_sequence_matches_scalar_gaussian_posterior() -> None:
    rng = np.random.default_rng(812)
    initial = rng.normal(size=(20_000, 1))

    def decode(parameters: np.ndarray) -> np.ndarray:
        return parameters.reshape(-1, 1, 1, 1)

    def evaluate(field: np.ndarray) -> ForwardResult:
        value = float(field.item())
        return ForwardResult("ok", (value,), value, 0.0, 0, "")

    stages = run_egg_esmda(
        initial,
        decode=decode,
        evaluator=evaluate,
        observation=np.asarray([1.0]),
        observation_covariance=np.asarray([[1.0]]),
        inflations=(1.0,),
        rng=np.random.default_rng(99),
        workers=1,
        svd_energy=1.0,
    )

    assert len(stages) == 2
    assert stages[-1].parameters[:, 0].mean() == pytest.approx(0.5, abs=0.02)
    assert stages[-1].parameters[:, 0].var(ddof=1) == pytest.approx(0.5, abs=0.02)
