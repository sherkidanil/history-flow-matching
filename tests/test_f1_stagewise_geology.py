from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np

repository = Path(__file__).parents[1]
sys.path.insert(0, str(repository / "scripts"))

from f1_stagewise_geology import _build_rows, _plot  # noqa: E402


def _write_inversion(path: Path, method: str, stage_predictions: tuple[float, ...]) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["method"] = method
        handle.attrs["parameterization_label"] = method
        handle.attrs["strategy"] = "augmentation"
        handle.create_dataset("observation", data=np.array([0.0]))
        handle.create_dataset("sigma", data=np.array([1.0]))
        for stage, prediction in enumerate(stage_predictions):
            group = handle.create_group(f"stage_{stage}")
            group.create_dataset(
                "logk",
                data=np.array(
                    [
                        [[[1.0, 1.0]]],
                        [[[1.0, 0.0]]],
                    ]
                ),
            )
            group.create_dataset("simulated_data", data=np.full((2, 1), prediction))
            group.create_dataset("fopt", data=np.array([4.0, 6.0]))


def _write_report(path: Path, method: str, artifact: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "method": method,
                "strategy": "augmentation",
                "truth_terminal_fopt": 5.0,
                "artifact": {
                    "path": artifact.as_posix(),
                    "git_commit": "a" * 40,
                },
            }
        ),
        encoding="utf-8",
    )


def test_build_rows_emits_all_stages_and_matched_misfit_control(tmp_path: Path) -> None:
    inversions: dict[str, Path] = {}
    reports: dict[str, Path] = {}
    for method, predictions in {
        "raw": (3.0, 2.0),
        "pca": (2.0, 1.0),
        "fm": (3.0, 2.0),
    }.items():
        inversion = tmp_path / f"{method}.h5"
        report = tmp_path / f"{method}.json"
        _write_inversion(inversion, method, predictions)
        _write_report(report, method, inversion)
        inversions[method] = inversion
        reports[method] = report

    stagewise, matched = _build_rows(
        inversions,
        reports=reports,
        truth=np.ones((1, 1, 1, 2)),
        active=np.ones((1, 1, 2), dtype=bool),
        threshold=0.5,
        wells={"INJECT1": (0, 0), "PROD1": (0, 1)},
        derived_git_commit="b" * 40,
    )

    assert len(stagewise) == 6
    assert len(matched) == 3
    assert list(stagewise[0]) == [
        "strategy",
        "parameterization",
        "stage",
        "mean_normalized_data_misfit",
        "bimodality_coefficient",
        "connectivity_mae_to_truth",
        "largest_component_fraction_p50",
        "fopt_p10",
        "fopt_p50",
        "fopt_p90",
        "covered",
        "source_artifact",
        "source_artifact_sha256",
        "source_report",
        "source_report_sha256",
        "source_git_commit",
        "derived_git_commit",
    ]
    selected = {row["parameterization"]: row for row in matched}
    assert selected["fm"]["stage"] == 1
    assert selected["fm"]["misfit_gap"] == 0.0
    assert selected["raw"]["stage"] == 1
    assert selected["pca"]["stage"] == 0
    assert all("diagnostic" in str(row["comparison_scope"]) for row in matched)


def test_tradeoff_plot_is_deterministic_vector_output(tmp_path: Path) -> None:
    rows = [
        {
            "parameterization": method,
            "stage": stage,
            "mean_normalized_data_misfit": 10.0 - stage,
            "bimodality_coefficient": 0.4 + 0.01 * stage,
            "connectivity_mae_to_truth": 0.3 - 0.01 * stage,
            "source_artifact_sha256": method * 8,
        }
        for method in ("raw", "pca", "fm")
        for stage in (0, 1)
    ]
    svg = tmp_path / "tradeoff.svg"
    pdf = tmp_path / "tradeoff.pdf"

    _plot(rows, svg, pdf)
    first = (svg.read_bytes(), pdf.read_bytes())
    _plot(rows, svg, pdf)

    assert svg.read_bytes() == first[0]
    assert pdf.read_bytes() == first[1]
    assert first[0].startswith(b"<?xml")
    assert first[1].startswith(b"%PDF")
