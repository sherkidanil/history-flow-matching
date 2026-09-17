from __future__ import annotations

import sys
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest

repository = Path(__file__).parents[1]
sys.path.insert(0, str(repository / "scripts"))

from f2_linearity_diagnostics import (  # noqa: E402
    _build_diagnostics,
    _cross_validated_linear_response,
)


def _write_artifact(path: Path, method: str) -> None:
    parameters = np.arange(1.0, 5.0)[:, None]
    decode = np.square if method == "fm" else lambda values: values
    with h5py.File(path, "w") as handle:
        handle.attrs["method"] = method
        handle.attrs["parameterization_label"] = method
        handle.attrs["strategy"] = "augmentation"
        for stage, scale in enumerate((1.0, 2.0) if method == "fm" else (1.0,)):
            group = handle.create_group(f"stage_{stage}")
            stage_parameters = scale * parameters
            group.create_dataset("parameters", data=stage_parameters)
            group.create_dataset("logk", data=decode(stage_parameters).reshape(4, 1, 1, 1))
            group.create_dataset("simulated_data", data=3.0 * stage_parameters)


def test_build_diagnostics_combines_linearity_midpoints_and_fm_shifts(
    tmp_path: Path,
) -> None:
    inversions: dict[str, Path] = {}
    for method in ("raw", "pca", "fm"):
        path = tmp_path / f"{method}.h5"
        _write_artifact(path, method)
        inversions[method] = path
    decoders = {
        "raw": lambda values: values.reshape(len(values), 1, 1, 1),
        "pca": lambda values: values.reshape(len(values), 1, 1, 1),
        "fm": lambda values: np.square(values).reshape(len(values), 1, 1, 1),
    }

    rows, payload = _build_diagnostics(
        inversions,
        decoders=decoders,
        svd_energy=1.0,
        pair_count=3,
        pair_seed=17,
        n_folds=2,
        fold_seed=23,
        derived_git_commit="a" * 40,
        decoder_metadata={"fm": {"checkpoint_sha256": "b" * 64}},
    )

    assert len(rows) == 5
    stage_zero = {row["parameterization"]: row for row in rows if row["stage"] == 0}
    assert set(stage_zero) == {"raw", "pca", "fm"}
    assert all(row["r2_lin"] == pytest.approx(1.0) for row in stage_zero.values())
    assert stage_zero["raw"]["midpoint_epsilon_p50"] == pytest.approx(0.0)
    assert stage_zero["pca"]["midpoint_epsilon_p50"] == pytest.approx(0.0)
    assert float(stage_zero["fm"]["midpoint_epsilon_p50"]) > 0.0
    update = next(row for row in rows if row["diagnostic"] == "update_shift")
    assert update["latent_relative_shift"] == pytest.approx(1.0)
    assert update["field_relative_shift"] == pytest.approx(3.0)
    assert update["field_to_latent_shift_ratio"] == pytest.approx(3.0)
    assert payload["pair_seed"] == 17
    assert payload["fold_seed"] == 23
    assert payload["n_folds"] == 2
    assert payload["parameterizations"]["fm"]["checkpoint_sha256"] == "b" * 64
    assert all(row["r2_oos_mean"] == pytest.approx(1.0) for row in stage_zero.values())
    final = next(row for row in rows if row["diagnostic"] == "final_stage_linearity_oos")
    assert final["parameterization"] == "fm"
    assert final["stage"] == 1


def test_cross_validated_linearity_exposes_in_sample_interpolation() -> None:
    rng = np.random.default_rng(91)
    parameters = rng.normal(size=(20, 19))
    simulated = rng.normal(size=(20, 4))

    scores = _cross_validated_linear_response(
        parameters,
        simulated,
        svd_energy=1.0,
        n_folds=5,
        seed=20260915,
    )

    assert scores.in_sample_r2 == pytest.approx(1.0, abs=1e-12)
    assert scores.mean < 0.5
    assert scores.std > 0.0
    assert len(scores.fold_scores) == 5
    assert scores == _cross_validated_linear_response(
        parameters,
        simulated,
        svd_energy=1.0,
        n_folds=5,
        seed=20260915,
    )
