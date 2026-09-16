from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

repository = Path(__file__).parents[1]
sys.path.insert(0, str(repository / "scripts"))

from m8_build_egg_inversion_tables import _cluster_row  # noqa: E402
from m8_plot_egg_inversion import _water_cut_quantiles  # noqa: E402
from m9_build_source_ablation import _loss_at_epoch  # noqa: E402


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
