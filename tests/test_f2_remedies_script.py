from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np

from fmgeo.artifacts import sha256_file

repository = Path(__file__).parents[1]
sys.path.insert(0, str(repository / "scripts"))
from f2_build_remedies import _build_rows  # noqa: E402

sys.path.pop(0)


def _write_inputs(root: Path, method: str) -> tuple[Path, Path]:
    artifact = root / f"{method}.h5"
    fields = np.asarray(
        [
            [[[0.0, 2.0]]],
            [[[0.0, 2.0]]],
            [[[2.0, 2.0]]],
            [[[0.0, 0.0]]],
        ]
    )
    with h5py.File(artifact, "w") as handle:
        handle.attrs["method"] = method
        handle.attrs["strategy"] = "augmentation"
        handle.attrs["parameterization_label"] = f"{method}_na8"
        handle.create_dataset("observation", data=[1.0])
        handle.create_dataset("truth_data", data=[1.0])
        handle.create_dataset("sigma", data=[1.0])
        group = handle.create_group("stage_8")
        group.create_dataset("logk", data=fields)
        group.create_dataset("simulated_data", data=np.arange(4.0)[:, None])
        group.create_dataset("fopt", data=[1.0, 2.0, 3.0, 4.0])
    report = root / f"{method}.json"
    report.write_text(
        json.dumps(
            {
                "method": method,
                "strategy": "augmentation",
                "remedy": "assimilation_steps",
                "n_assimilations": 8,
                "localization_enabled": False,
                "localization_radius_m": None,
                "latent_rank": None,
                "n_sim": 900,
                "n_failed": 0,
                "truth_terminal_fopt": 2.5,
                "config_hash": "a" * 64,
                "artifact": {
                    "path": artifact.as_posix(),
                    "sha256": sha256_file(artifact),
                    "git_commit": "c" * 40,
                },
                "stages": [
                    {
                        "mean_field_spread": 0.5,
                        "relative_field_shift_from_prior": 0.25,
                        "effective_ensemble_members": 3.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return artifact, report


def test_build_remedy_rows_combines_science_collapse_and_provenance(
    tmp_path: Path,
) -> None:
    inputs = {method: _write_inputs(tmp_path, method) for method in ("raw", "pca", "fm")}

    rows = _build_rows(
        {method: values[0] for method, values in inputs.items()},
        reports={method: values[1] for method, values in inputs.items()},
        truth=np.asarray([[[[0.0, 2.0]]]]),
        active=np.ones((1, 1, 2), dtype=bool),
        threshold=1.0,
        wells={"INJECT1": (0, 0), "PROD1": (0, 1)},
        derived_git_commit="d" * 40,
    )

    assert len(rows) == 3
    assert {row["parameterization"] for row in rows} == {"raw", "pca", "fm"}
    assert all(row["remedy"] == "assimilation_steps" for row in rows)
    assert all(row["n_assimilations"] == 8 for row in rows)
    assert all(row["ensemble_size"] == 4 for row in rows)
    assert all(row["effective_ensemble_members"] == 3.0 for row in rows)
    assert all(row["source_git_commit"] == "c" * 40 for row in rows)
    assert all(row["derived_git_commit"] == "d" * 40 for row in rows)


def test_build_remedy_rows_accepts_two_labeled_raw_variants(tmp_path: Path) -> None:
    inputs = {method: _write_inputs(tmp_path, method) for method in ("raw", "pca", "fm")}
    second_raw = tmp_path / "raw_unlocalized.h5"
    second_report = tmp_path / "raw_unlocalized.json"
    shutil.copy2(inputs["raw"][0], second_raw)
    shutil.copy2(inputs["raw"][1], second_report)
    inversions = {
        "raw_localized": inputs["raw"][0],
        "fm_localized": inputs["fm"][0],
        "pca_unlocalized": inputs["pca"][0],
        "raw_unlocalized": second_raw,
    }
    reports = {
        "raw_localized": inputs["raw"][1],
        "fm_localized": inputs["fm"][1],
        "pca_unlocalized": inputs["pca"][1],
        "raw_unlocalized": second_report,
    }

    rows = _build_rows(
        inversions,
        reports=reports,
        truth=np.asarray([[[[0.0, 2.0]]]]),
        active=np.ones((1, 1, 2), dtype=bool),
        threshold=1.0,
        wells={"INJECT1": (0, 0), "PROD1": (0, 1)},
        derived_git_commit="d" * 40,
    )

    assert {row["variant"] for row in rows} == set(inversions)
    assert [row["parameterization"] for row in rows].count("raw") == 2
