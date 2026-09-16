"""Build the Egg coarse-to-full resolution transfer table and variogram figure."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
from m8_egg_case import EGG_SHAPE, load_egg_ensemble, load_strategy_config

from fmgeo.artifacts import sha256_file
from fmgeo.forward.egg import parse_egg_well_locations
from fmgeo.metrics.egg import egg_well_connectivity
from fmgeo.metrics.geology import experimental_variogram
from fmgeo.resolution import average_pool_horizontal


def _physical_lags(count: int, *, cell_size: float) -> np.ndarray:
    if count < 1 or cell_size <= 0:
        raise ValueError("lag count and physical cell size must be positive")
    return np.arange(1, count + 1, dtype=np.float64) * cell_size


def _scaled_wells(
    wells: Mapping[str, Sequence[int]], *, factor: int
) -> dict[str, tuple[int, int]]:
    if factor < 1:
        raise ValueError("well scaling factor must be positive")
    return {
        name: (int(location[0]) // factor, int(location[1]) // factor)
        for name, location in wells.items()
    }


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _reference_at_shape(
    fields: np.ndarray, full_active: np.ndarray, shape: tuple[int, int, int]
) -> tuple[np.ndarray, np.ndarray, int]:
    if shape == EGG_SHAPE:
        return fields, full_active, 1
    if shape[0] != EGG_SHAPE[0] or EGG_SHAPE[1] % shape[1] or EGG_SHAPE[2] % shape[2]:
        raise ValueError("evaluation shape is not an Egg horizontal coarsening")
    factors = (EGG_SHAPE[1] // shape[1], EGG_SHAPE[2] // shape[2])
    if factors[0] != factors[1]:
        raise ValueError("Egg horizontal coarsening must be isotropic")
    pooled, active = average_pool_horizontal(fields, active_mask=full_active, factor=factors[0])
    return pooled, active, factors[0]


def _plot_variograms(
    path: Path,
    curves: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]],
    reference: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    colors = {"white": "#440154", "matern": "#21918c"}
    for row, architecture in enumerate(("unet3d", "uno3d")):
        for column, direction in enumerate(("x", "y")):
            axis = axes[row, column]
            reference_lags, reference_values = reference[direction]
            axis.plot(
                reference_lags,
                reference_values,
                color="black",
                linewidth=2,
                label="held-out",
            )
            for source in ("white", "matern"):
                for training_resolution, linestyle in (("coarse", "--"), ("full", "-")):
                    key = (architecture, source, training_resolution, direction)
                    lags, values = curves[key]
                    axis.plot(
                        lags,
                        values,
                        color=colors[source],
                        linestyle=linestyle,
                        label=f"{source}, train {training_resolution}",
                    )
            axis.set(
                title=f"{architecture.removesuffix('3d').upper()} variogram {direction.upper()}",
                xlabel="Physical horizontal lag (fine-cell units)",
                ylabel="Semivariance",
            )
            axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="outside lower center", ncol=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, format="svg")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy-config", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--realizations-dir", type=Path, required=True)
    parser.add_argument("--evaluation-report", action="append", type=Path, required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    args = parser.parse_args()

    strategy = load_strategy_config(args.strategy_config)
    ensemble = load_egg_ensemble(args.realizations_dir)
    heldout = np.asarray([index - 1 for index in strategy.heldout_indices])
    reference_full = ensemble[heldout]
    full_active = np.any(ensemble != 0.0, axis=0)
    wells_full = parse_egg_well_locations(args.deck.read_text(encoding="utf-8", errors="replace"))
    derived_git_commit = _git_commit()
    rows: list[dict[str, object]] = []
    curves: dict[tuple[str, str, str, str], tuple[np.ndarray, np.ndarray]] = {}

    for report_path in args.evaluation_report:
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        architecture = str(report["model_kind"])
        source = str(report["source_kind"])
        training_shape = tuple(int(value) for value in report["training_shape_zyx"])
        evaluation_shape = tuple(int(value) for value in report["evaluation_shape_zyx"])
        if len(training_shape) != 3 or len(evaluation_shape) != 3:
            raise ValueError("resolution reports must contain three-dimensional shapes")
        training_resolution = "full" if training_shape == EGG_SHAPE else "coarse"
        evaluation_resolution = "full" if evaluation_shape == EGG_SHAPE else "coarse"
        artifact_path = Path(report["artifact"]["path"])
        artifact_hash = sha256_file(artifact_path)
        if artifact_hash != report["artifact"]["sha256"]:
            raise ValueError(f"artifact hash mismatch for {report_path}")
        with h5py.File(artifact_path) as handle:
            generated = np.asarray(handle["logk"], dtype=np.float64)
            active = np.asarray(handle["active_mask"], dtype=bool)
        reference, reference_active, factor = _reference_at_shape(
            reference_full, full_active, evaluation_shape
        )
        if not np.array_equal(active, reference_active):
            raise ValueError("generated and official active masks differ")
        wells = _scaled_wells(wells_full, factor=factor)
        connectivity = egg_well_connectivity(
            generated, threshold=float(report["high_permeability_threshold_logk"]), wells=wells
        )
        reference_connectivity = egg_well_connectivity(
            reference, threshold=float(report["high_permeability_threshold_logk"]), wells=wells
        )
        metrics = report["metrics"]
        rows.append(
            {
                "architecture": architecture,
                "source": source,
                "training_resolution": training_resolution,
                "evaluation_resolution": evaluation_resolution,
                "training_shape_zyx": "x".join(str(value) for value in training_shape),
                "evaluation_shape_zyx": "x".join(str(value) for value in evaluation_shape),
                "sample_count": report["sample_count"],
                "integration_steps": report["integration_steps"],
                "roundtrip_relative_error": report["roundtrip_relative_error"],
                "marginal_ks": metrics["marginal_ks"],
                "variogram_x_nrmse": metrics["variogram_x_nrmse"],
                "variogram_y_nrmse": metrics["variogram_y_nrmse"],
                "spanning_fraction_abs_error": metrics["spanning_fraction_abs_error"],
                "bimodality_abs_error": metrics["bimodality_abs_error"],
                "connectivity_mae": float(
                    np.mean(
                        [
                            abs(connectivity[pair] - reference_connectivity[pair])
                            for pair in connectivity
                        ]
                    )
                ),
                "source_corr_len_cells_zyx": "x".join(
                    str(value) for value in report["source_corr_len_cells_zyx"]
                ),
                "source_report": report_path.as_posix(),
                "source_report_sha256": sha256_file(report_path),
                "source_artifact": artifact_path.as_posix(),
                "source_artifact_sha256": artifact_hash,
                "source_git_commit": report["artifact"]["git_commit"],
                "derived_git_commit": derived_git_commit,
            }
        )
        if evaluation_resolution == "full":
            lag_count = int(report["max_variogram_lag"])
            lags = _physical_lags(lag_count, cell_size=1.0)
            for direction, axis in (("x", 2), ("y", 1)):
                curves[(architecture, source, training_resolution, direction)] = (
                    lags,
                    experimental_variogram(
                        generated[:128], axis=axis, max_lag=lag_count, active_mask=active
                    ),
                )

    expected = {
        (architecture, source, training, evaluation)
        for architecture in ("unet3d", "uno3d")
        for source in ("white", "matern")
        for training, evaluation in (("coarse", "coarse"), ("coarse", "full"), ("full", "full"))
    }
    actual = {
        (
            str(row["architecture"]),
            str(row["source"]),
            str(row["training_resolution"]),
            str(row["evaluation_resolution"]),
        )
        for row in rows
    }
    if actual != expected:
        raise ValueError("evaluation reports do not form the complete 12-run resolution matrix")
    rows.sort(
        key=lambda row: (
            str(row["architecture"]),
            str(row["source"]),
            str(row["training_resolution"]),
            str(row["evaluation_resolution"]),
        )
    )
    reference_lag_count = max(int(report["max_variogram_lag"]) for report in (
        json.loads(path.read_text(encoding="utf-8")) for path in args.evaluation_report
    ) if tuple(report["evaluation_shape_zyx"]) == EGG_SHAPE)
    reference_lags = _physical_lags(reference_lag_count, cell_size=1.0)
    reference_curves = {
        direction: (
            reference_lags,
            experimental_variogram(
                reference_full,
                axis=axis,
                max_lag=reference_lag_count,
                active_mask=full_active,
            ),
        )
        for direction, axis in (("x", 2), ("y", 1))
    }
    _write_csv(args.table_output, rows)
    plt.rcParams["svg.hashsalt"] = "fmgeo-resolution-ablation"
    _plot_variograms(args.figure_output, curves, reference_curves)
    print(json.dumps({"rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
