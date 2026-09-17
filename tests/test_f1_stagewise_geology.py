from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np

repository = Path(__file__).parents[1]
sys.path.insert(0, str(repository / "scripts"))

from f1_stagewise_geology import _build_rows, _plot  # noqa: E402


def _write_inversion(
    path: Path,
    method: str,
    label: str,
    stage_predictions: tuple[float, ...],
) -> None:
    with h5py.File(path, "w") as handle:
        handle.attrs["method"] = method
        handle.attrs["parameterization_label"] = label
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


def test_build_rows_emits_all_remedy_stages_and_bracketing_misfits(
    tmp_path: Path,
) -> None:
    inversions: dict[str, Path] = {}
    reports: dict[str, Path] = {}
    variants = {
        "raw_na8": ("raw", (4.0, 2.0)),
        "pca_na8": ("pca", (4.0, 1.0)),
        "fm_na8": ("fm", (5.0, 3.0)),
        "fm_na16": ("fm", (6.0, 4.0)),
        "fm_na8_n200": ("fm", (7.0, 5.0)),
    }
    for label, (method, predictions) in variants.items():
        inversion = tmp_path / f"{label}.h5"
        report = tmp_path / f"{label}.json"
        _write_inversion(inversion, method, label, predictions)
        _write_report(report, method, inversion)
        inversions[label] = inversion
        reports[label] = report

    stagewise, matched = _build_rows(
        inversions,
        reports=reports,
        truth=np.ones((1, 1, 1, 2)),
        active=np.ones((1, 1, 2), dtype=bool),
        threshold=0.5,
        wells={"INJECT1": (0, 0), "PROD1": (0, 1)},
        derived_git_commit="b" * 40,
    )

    assert len(stagewise) == 10
    assert len(matched) == 5
    assert list(stagewise[0]) == [
        "strategy",
        "variant",
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
    selected = [(row["variant"], row["stage"]) for row in matched]
    assert selected == [
        ("fm_na8", 1),
        ("raw_na8", 0),
        ("raw_na8", 1),
        ("pca_na8", 0),
        ("pca_na8", 1),
    ]
    assert matched[0]["misfit_gap"] == 0.0
    assert [row["misfit_relation"] for row in matched] == [
        "target",
        "above",
        "below",
        "above",
        "below",
    ]
    assert all("diagnostic" in str(row["comparison_scope"]) for row in matched)


def test_build_rows_uses_one_exact_stage_instead_of_duplicate_brackets(
    tmp_path: Path,
) -> None:
    inversions: dict[str, Path] = {}
    reports: dict[str, Path] = {}
    for label, (method, predictions) in {
        "raw_na8": ("raw", (4.0, 3.0)),
        "pca_na8": ("pca", (4.0, 3.0)),
        "fm_na8": ("fm", (5.0, 3.0)),
        "fm_na16": ("fm", (6.0, 4.0)),
        "fm_na8_n200": ("fm", (7.0, 5.0)),
    }.items():
        inversion = tmp_path / f"{label}.h5"
        report = tmp_path / f"{label}.json"
        _write_inversion(inversion, method, label, predictions)
        _write_report(report, method, inversion)
        inversions[label] = inversion
        reports[label] = report

    stagewise, matched = _build_rows(
        inversions,
        reports=reports,
        truth=np.ones((1, 1, 1, 2)),
        active=np.ones((1, 1, 2), dtype=bool),
        threshold=0.5,
        wells={"INJECT1": (0, 0), "PROD1": (0, 1)},
        derived_git_commit="b" * 40,
    )

    assert len(stagewise) == 10
    assert len(matched) == 3
    assert [(row["variant"], row["stage"]) for row in matched] == [
        ("fm_na8", 1),
        ("raw_na8", 1),
        ("pca_na8", 1),
    ]
    assert all(row["misfit_relation"] == "target" for row in matched)


def test_tradeoff_plot_is_deterministic_vector_output(tmp_path: Path) -> None:
    rows = [
        {
            "variant": variant,
            "parameterization": method,
            "stage": stage,
            "mean_normalized_data_misfit": 10.0 - stage,
            "bimodality_coefficient": 0.4 + 0.01 * stage,
            "connectivity_mae_to_truth": 0.3 - 0.01 * stage,
            "source_artifact_sha256": method * 8,
        }
        for variant, method in (
            ("raw_na8", "raw"),
            ("pca_na8", "pca"),
            ("fm_na8", "fm"),
            ("fm_na16", "fm"),
            ("fm_na8_n200", "fm"),
        )
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
