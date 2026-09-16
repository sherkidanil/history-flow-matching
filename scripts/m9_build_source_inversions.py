"""Build the controlled Egg FM source-inversion comparison."""

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
from m8_egg_case import load_egg_ensemble

from fmgeo.artifacts import sha256_file
from fmgeo.forward.egg import parse_egg_well_locations
from fmgeo.inverse.egg import load_egg_inversion_config
from fmgeo.metrics.egg import (
    bimodality_coefficient,
    egg_cluster_size_summary,
    egg_well_connectivity,
)
from fmgeo.metrics.uq import crps_ensemble, energy_score, interval_width


def _labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    return label, Path(raw_path)


def _last_stage(handle: h5py.File) -> str:
    stages = sorted(
        (int(name.removeprefix("stage_")), name)
        for name in handle
        if name.startswith("stage_")
    )
    if not stages:
        raise ValueError("inversion artifact contains no stages")
    return stages[-1][1]


def _comparison_metrics(
    fields: np.ndarray,
    *,
    truth: np.ndarray,
    active: np.ndarray,
    threshold: float,
    wells: Mapping[str, Sequence[int]],
) -> dict[str, float | int]:
    connectivity = egg_well_connectivity(fields, threshold=threshold, wells=wells)
    truth_connectivity = egg_well_connectivity(truth, threshold=threshold, wells=wells)
    cluster = egg_cluster_size_summary(fields >= threshold)
    truth_cluster = egg_cluster_size_summary(truth >= threshold)
    return {
        "connectivity_mae_to_truth": float(
            np.mean([abs(connectivity[pair] - truth_connectivity[pair]) for pair in connectivity])
        ),
        "bimodality_coefficient": bimodality_coefficient(fields[:, active]),
        "truth_bimodality_coefficient": bimodality_coefficient(truth[:, active]),
        **cluster,
        **{f"truth_{name}": value for name, value in truth_cluster.items()},
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError("expected a numeric table value")
    return float(value)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def _plot(path: Path, rows: list[dict[str, object]]) -> None:
    labels = [str(row["variant"]) for row in rows]
    colors = ["#440154", "#21918c", "#fde725"]
    x = np.arange(len(rows))
    figure, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)

    lower = np.asarray([_as_float(row["fopt_p10"]) for row in rows])
    center = np.asarray([_as_float(row["fopt_p50"]) for row in rows])
    upper = np.asarray([_as_float(row["fopt_p90"]) for row in rows])
    for index, color in enumerate(colors):
        axes[0, 0].errorbar(
            x[index],
            center[index],
            yerr=np.asarray(
                [[center[index] - lower[index]], [upper[index] - center[index]]]
            ),
            fmt="none",
            ecolor=color,
            elinewidth=5,
            capsize=6,
        )
    axes[0, 0].scatter(x, center, c=colors, zorder=3)
    axes[0, 0].axhline(_as_float(rows[0]["truth_fopt"]), color="black", linestyle="--")
    axes[0, 0].set(title="Terminal FOPT P10–P90", ylabel="FOPT")

    panels = (
        ("mean_normalized_data_misfit", "Normalized data misfit", None),
        ("connectivity_mae_to_truth", "Connectivity MAE", None),
        ("bimodality_abs_error", "Bimodality absolute error", None),
        ("largest_component_fraction_abs_error", "Largest-cluster fraction error", None),
        ("component_size_p50_abs_error", "Median cluster-size error", None),
    )
    for axis, (key, title, scale) in zip(axes.flat[1:], panels, strict=True):
        axis.bar(x, [_as_float(row[key]) for row in rows], color=colors)
        axis.set(title=title, yscale=scale or "linear")
    for axis in axes.flat:
        axis.set_xticks(x, labels, rotation=15)
        axis.grid(axis="y", alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        format="svg",
        metadata={"Creator": "fmgeo m9_build_source_inversions.py", "Date": None},
    )
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--realizations-dir", type=Path, required=True)
    parser.add_argument("--active-source", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--inversion", action="append", type=_labeled_path, required=True)
    parser.add_argument("--inversion-report", action="append", type=_labeled_path, required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    args = parser.parse_args()

    inversions = dict(args.inversion)
    reports = dict(args.inversion_report)
    expected = {"white", "matern", "matern_misspec"}
    if set(inversions) != expected or set(reports) != expected:
        raise ValueError("source inversion comparison requires white, matern, and matern_misspec")

    config = load_egg_inversion_config(args.config)
    evaluation = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    threshold = float(evaluation["high_permeability_threshold_logk"])
    with h5py.File(args.active_source) as handle:
        active = np.asarray(handle["active_mask"], dtype=bool)
    reference = load_egg_ensemble(args.realizations_dir)
    truth = reference[config.truth_realization - 1 : config.truth_realization]
    wells = parse_egg_well_locations(args.deck.read_text(encoding="utf-8", errors="replace"))
    derived_git_commit = _git_commit()

    rows: list[dict[str, object]] = []
    for label in ("white", "matern", "matern_misspec"):
        artifact_path = inversions[label]
        artifact_hash = sha256_file(artifact_path)
        report_path = reports[label]
        report: dict[str, Any] = json.loads(report_path.read_text(encoding="utf-8"))
        with h5py.File(artifact_path) as handle:
            if str(handle.attrs["method"]) != "fm":
                raise ValueError(f"{label} inversion is not an FM artifact")
            if str(handle.attrs["parameterization_label"]) != label:
                raise ValueError(f"{label} inversion label does not match its artifact")
            if str(handle.attrs["config_hash"]) != str(report["config_hash"]):
                raise ValueError(f"{label} inversion/report configuration mismatch")
            final_stage = handle[_last_stage(handle)]
            fields = np.asarray(final_stage["logk"], dtype=np.float64)
            fopt_ensemble = np.asarray(final_stage["fopt"], dtype=np.float64)
            simulated_data = np.asarray(final_stage["simulated_data"], dtype=np.float64)
            truth_data = np.asarray(handle["truth_data"], dtype=np.float64)
            sigma = np.asarray(handle["sigma"], dtype=np.float64)
        if report.get("parameterization_label") != label:
            raise ValueError(f"{label} report label does not match")
        if report["artifact"]["sha256"] != artifact_hash:
            raise ValueError(f"{label} report artifact hash does not match")
        final = report["stages"][-1]
        fopt = final["fopt"]
        metrics = _comparison_metrics(
            fields, truth=truth, active=active, threshold=threshold, wells=wells
        )
        rows.append(
            {
                "variant": label,
                "fopt_p10": fopt["P10"],
                "fopt_p50": fopt["P50"],
                "fopt_p90": fopt["P90"],
                "truth_fopt": fopt["truth"],
                "fopt_covered": fopt["covered"],
                "fopt_p10_p90_width": interval_width(fopt["P10"], fopt["P90"]),
                "fopt_crps": crps_ensemble(fopt_ensemble, observation=fopt["truth"]),
                "normalized_observation_energy_score": energy_score(
                    simulated_data / sigma[None, :],
                    observation=truth_data / sigma,
                ),
                "mean_normalized_data_misfit": final["mean_normalized_data_misfit"],
                "n_sim": report["n_sim"],
                "n_failed": report["n_failed"],
                **metrics,
                "bimodality_abs_error": abs(
                    float(metrics["bimodality_coefficient"])
                    - float(metrics["truth_bimodality_coefficient"])
                ),
                "largest_component_fraction_abs_error": abs(
                    float(metrics["largest_component_fraction_p50"])
                    - float(metrics["truth_largest_component_fraction_p50"])
                ),
                "component_size_p50_abs_error": abs(
                    float(metrics["component_size_p50"])
                    - float(metrics["truth_component_size_p50"])
                ),
                "checkpoint_sha256": report["parameterization"]["checkpoint_sha256"],
                "source_report": report_path.as_posix(),
                "source_report_sha256": sha256_file(report_path),
                "source_artifact": artifact_path.as_posix(),
                "source_artifact_sha256": artifact_hash,
                "source_git_commit": report["artifact"]["git_commit"],
                "derived_git_commit": derived_git_commit,
            }
        )
    _write_csv(args.table_output, rows)
    plt.rcParams["svg.hashsalt"] = "fmgeo-source-inversions"
    _plot(args.figure_output, rows)
    print(json.dumps({"rows": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
