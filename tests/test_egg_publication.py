from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

repository = Path(__file__).parents[1]
sys.path.insert(0, str(repository / "scripts"))

from m8_build_egg_inversion_tables import _cluster_row  # noqa: E402
from m8_plot_egg_inversion import _water_cut_quantiles  # noqa: E402
from m9_build_resolution_ablation import _physical_lags, _scaled_wells  # noqa: E402
from m9_build_source_ablation import _loss_at_epoch  # noqa: E402
from m9_build_source_inversions import _comparison_metrics, _plot  # noqa: E402


def test_cluster_row_contains_measured_summary_and_provenance() -> None:
    fields = np.zeros((1, 1, 2, 3), dtype=float)
    fields[0, 0, 0, :2] = 2.0

    row = _cluster_row(
        strategy="augmentation",
        category="raw",
        fields=fields,
        threshold=1.0,
        source=Path("artifacts/raw.h5"),
        source_hash="a" * 64,
        derived_git_commit="b" * 40,
    )

    assert row["component_count"] == 1
    assert row["component_size_p50"] == 2.0
    assert row["largest_component_fraction_p50"] == 1.0
    assert row["source_artifact"] == "artifacts/raw.h5"
    assert row["source_artifact_sha256"] == "a" * 64
    assert row["derived_git_commit"] == "b" * 40


def test_water_cut_quantiles_preserve_times_and_member_axis() -> None:
    metadata = [
        {"forecast_times_days": [0.0, 30.0], "water_cut": {"PROD1": [0.0, 0.2]}},
        {"forecast_times_days": [0.0, 30.0], "water_cut": {"PROD1": [0.0, 0.8]}},
    ]

    times, quantiles = _water_cut_quantiles(metadata, "PROD1")

    np.testing.assert_array_equal(times, [0.0, 30.0])
    np.testing.assert_allclose(quantiles[:, 1], [0.26, 0.5, 0.74])


def test_ablation_epoch_loss_uses_complete_batches_per_epoch() -> None:
    history = [float(value) for value in range(12)]

    assert _loss_at_epoch(history, epoch=1, training_samples=5, batch_size=2) == 2.0
    assert _loss_at_epoch(history, epoch=3, training_samples=5, batch_size=2) == 8.0


def test_source_inversion_metrics_are_exact_for_truth_ensemble() -> None:
    fields = np.asarray(
        [[[[0.0, 2.0, 2.0], [0.0, 2.0, 2.0]], [[0.0, 2.0, 2.0], [0.0, 2.0, 2.0]]]]
    )
    metrics = _comparison_metrics(
        fields,
        truth=fields,
        active=np.ones((2, 2, 3), dtype=bool),
        threshold=1.0,
        wells={"INJECT1": (0, 1), "PROD1": (1, 2)},
    )

    assert metrics["connectivity_mae_to_truth"] == 0.0
    assert metrics["bimodality_coefficient"] == metrics["truth_bimodality_coefficient"]
    assert metrics["largest_component_fraction_p50"] == 1.0


def test_source_inversion_plot_renders_three_colored_intervals(tmp_path: Path) -> None:
    rows = [
        {
            "variant": variant,
            "fopt_p10": 1.0 + index,
            "fopt_p50": 2.0 + index,
            "fopt_p90": 3.0 + index,
            "truth_fopt": 2.5,
            "mean_normalized_data_misfit": 1.0,
            "connectivity_mae_to_truth": 0.1,
            "bimodality_abs_error": 0.2,
            "largest_component_fraction_abs_error": 0.3,
            "component_size_p50_abs_error": 4.0,
        }
        for index, variant in enumerate(("white", "matern", "matern_misspec"))
    ]
    output = tmp_path / "comparison.svg"

    _plot(output, rows)

    assert output.read_text(encoding="utf-8").startswith("<?xml")


def test_resolution_publication_uses_physical_lags_and_scaled_wells() -> None:
    np.testing.assert_array_equal(_physical_lags(4, cell_size=2.0), [2.0, 4.0, 6.0, 8.0])
    assert _scaled_wells({"INJECT1": (5, 9), "PROD1": (10, 12)}, factor=2) == {
        "INJECT1": (2, 4),
        "PROD1": (5, 6),
    }
