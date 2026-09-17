from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
import pytest

repository = Path(__file__).parents[1]
sys.path.insert(0, str(repository / "scripts"))

from g2_anomaly_diagnostics import _build_rows  # noqa: E402


def _write_artifact(
    path: Path, label: str, ensemble_size: int, n_assimilations: int
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["method"] = "fm"
        handle.attrs["parameterization_label"] = label
        handle.attrs["strategy"] = "augmentation"
        handle.create_dataset("observation", data=np.zeros(2))
        handle.create_dataset("sigma", data=np.ones(2))
        parameters = np.arange(ensemble_size * 2, dtype=float).reshape(ensemble_size, 2)
        fields = parameters.reshape(ensemble_size, 1, 1, 2)
        simulated = np.column_stack((parameters[:, 0], 2.0 * parameters[:, 1]))
        for stage in range(n_assimilations + 1):
            scale = 1.0 + 0.1 * stage
            group = handle.create_group(f"stage_{stage}")
            group.create_dataset("parameters", data=scale * parameters)
            group.create_dataset("logk", data=scale * fields)
            group.create_dataset("simulated_data", data=scale * simulated)
            if label != "fm":
                group.attrs["effective_ensemble_members"] = ensemble_size - stage


def _write_report(path: Path, artifact: Path, label: str, n_assimilations: int) -> None:
    path.write_text(
        json.dumps(
            {
                "method": "fm",
                "parameterization_label": label,
                "strategy": "augmentation",
                "prior_sha256": "a" * 64,
                "truth_case": "truth/EGG",
                "observation_count": 2,
                "observation_noise_seed": 17,
                "n_assimilations": n_assimilations,
                "parameterization": {
                    "checkpoint_sha256": "b" * 64,
                    "fm_config_hash": "c" * 64,
                    "training_data_sha256": "d" * 64,
                },
                "artifact": {
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "git_commit": "e" * 40,
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_rows_measures_stagewise_anomaly_mechanisms(tmp_path: Path) -> None:
    inversions: dict[str, Path] = {}
    reports: dict[str, Path] = {}
    inflations: dict[str, tuple[float, ...]] = {}
    for label, size, n_assimilations in (
        ("fm", 4, 4),
        ("fm_na8", 4, 8),
        ("fm_na16", 4, 16),
        ("fm_na8_n200", 6, 8),
    ):
        artifact = tmp_path / f"{label}.h5"
        report = tmp_path / f"{label}.json"
        _write_artifact(artifact, label, size, n_assimilations)
        _write_report(report, artifact, label, n_assimilations)
        inversions[label] = artifact
        reports[label] = report
        inflations[label] = (float(n_assimilations),) * n_assimilations

    rows = _build_rows(
        inversions,
        reports=reports,
        inflations=inflations,
        derived_git_commit="f" * 40,
    )

    assert len(rows) == 40
    assert {row["variant"] for row in rows} == set(inversions)
    assert all(row["inflation_reciprocal_sum"] == pytest.approx(1.0) for row in rows)
    assert all(row["configuration_consistent"] is True for row in rows)
    initial = [row for row in rows if row["stage"] == 0]
    updated = [row for row in rows if row["stage"] == 1]
    assert all(row["field_step_relative_shift"] == 0.0 for row in initial)
    assert all(float(row["field_step_relative_shift"]) > 0.0 for row in updated)
    assert all(row["distance_from_stage0"] == pytest.approx(0.1) for row in updated)
    assert all(0.0 < float(row["ensemble_data_variance_fraction"]) < 1.0 for row in rows)
    assert all(float(row["condition_number"]) >= 1.0 for row in rows)
    assert all(np.isfinite(float(row["effective_ensemble_members"])) for row in rows)
    n200 = [row for row in rows if row["variant"] == "fm_na8_n200"]
    assert all(row["stage0_prefix_matches_fm_na8"] is True for row in n200)
    assert all(row["stage0_parameter_count"] == 6 for row in n200)


def test_build_rows_rejects_core_provenance_mismatch(tmp_path: Path) -> None:
    inversions: dict[str, Path] = {}
    reports: dict[str, Path] = {}
    inflations: dict[str, tuple[float, ...]] = {}
    for label in ("fm", "fm_na8", "fm_na16", "fm_na8_n200"):
        artifact = tmp_path / f"{label}.h5"
        report = tmp_path / f"{label}.json"
        _write_artifact(artifact, label, 4, 4)
        _write_report(report, artifact, label, 4)
        inversions[label] = artifact
        reports[label] = report
        inflations[label] = (4.0,) * 4
    payload = json.loads(reports["fm_na16"].read_text(encoding="utf-8"))
    payload["observation_noise_seed"] = 99
    reports["fm_na16"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="observation_noise_seed"):
        _build_rows(
            inversions,
            reports=reports,
            inflations=inflations,
            derived_git_commit="f" * 40,
        )
