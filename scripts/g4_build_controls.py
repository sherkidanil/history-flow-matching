"""Build the final bounded G4 Egg comparison table and trajectory figure."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import h5py  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
from f2_build_remedies import _build_rows, _load_report, _write_csv
from f3_build_inversion_v2 import _parse_labeled_paths
from m8_egg_case import load_egg_ensemble

from fmgeo.forward.egg import EGG_PRODUCERS, parse_egg_well_locations
from fmgeo.inverse.egg import load_egg_inversion_config
from fmgeo.plotting import save_vector_figure


@dataclass(frozen=True)
class G4VariantSpec:
    method: str
    ensemble_size: int
    bounded: bool
    localized: bool
    color: str
    label: str


G4_SPECS = {
    "raw_na8": G4VariantSpec("raw", 100, False, False, "#440154", "Raw, N=100"),
    "pca_na8": G4VariantSpec("pca", 100, False, False, "#21918c", "PCA, N=100"),
    "fm_na8": G4VariantSpec("fm", 100, False, False, "#d49b00", "FM, N=100"),
    "raw_na8_bounded": G4VariantSpec(
        "raw", 100, True, False, "#482878", "Raw bounded, N=100"
    ),
    "pca_na8_bounded": G4VariantSpec(
        "pca", 100, True, False, "#35b779", "PCA bounded, N=100"
    ),
    "fm_na8_bounded": G4VariantSpec(
        "fm", 100, True, False, "#f0c808", "FM bounded, N=100"
    ),
    "raw_na8_localized": G4VariantSpec(
        "raw", 100, True, True, "#7e1e9c", "Raw bounded + localized, N=100"
    ),
    "pca_na8_localized": G4VariantSpec(
        "pca", 100, True, True, "#00a087", "PCA bounded + localized, N=100"
    ),
    "fm_na8_localized": G4VariantSpec(
        "fm", 100, True, True, "#d95f02", "FM bounded + localized, N=100"
    ),
    "raw_na8_n200": G4VariantSpec(
        "raw", 200, True, False, "#31688e", "Raw bounded, N=200"
    ),
    "fm_na8_n200": G4VariantSpec(
        "fm", 200, True, False, "#e76f51", "FM bounded, N=200"
    ),
}
G4_VARIANTS = tuple(G4_SPECS)



class G4StageRow(TypedDict):
    variant: str
    stage: int
    mean_normalized_data_misfit: float
    clipped_fraction: float | None


def _stage_rows(reports: Mapping[str, Path]) -> list[G4StageRow]:
    """Load stage trajectories in the fixed G4 comparison order."""

    if tuple(reports) != G4_VARIANTS:
        raise ValueError(f"G4 reports must be ordered as: {', '.join(G4_VARIANTS)}")
    rows: list[G4StageRow] = []
    for variant, path in reports.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        spec = G4_SPECS[variant]
        if report.get("parameterization_label") != variant:
            raise ValueError(f"G4 report label does not match input variant: {variant}")
        if report.get("method") != spec.method:
            raise ValueError(f"G4 report method does not match variant: {variant}")
        if bool(report.get("localization_enabled")) != spec.localized:
            raise ValueError(f"G4 report localization does not match variant: {variant}")
        bounds = report.get("logk_bounds")
        if (bounds is not None) != spec.bounded:
            raise ValueError(f"G4 report bounds do not match variant: {variant}")
        if spec.bounded and tuple(float(value) for value in bounds) != (2.0, 11.0):
            raise ValueError(f"G4 report uses unexpected bounds: {variant}")
        stages = report["stages"]
        ensemble_size = report.get("ensemble_size")
        if ensemble_size is None:
            ensemble_size = int(report["n_sim"]) // len(stages)
        if int(ensemble_size) != spec.ensemble_size:
            raise ValueError(f"G4 report ensemble size does not match variant: {variant}")
        for stage in stages:
            clipped = stage.get("clipped_fraction")
            if spec.bounded and clipped is None:
                raise ValueError(f"bounded G4 report lacks clipping metadata: {variant}")
            rows.append(
                {
                    "variant": variant,
                    "stage": int(stage["stage"]),
                    "mean_normalized_data_misfit": float(stage["mean_normalized_data_misfit"]),
                    "clipped_fraction": None if not spec.bounded else float(clipped),
                }
            )
    return rows


def _plot(rows: list[G4StageRow], svg_path: Path, pdf_path: Path) -> None:
    """Plot G4 data-fit and clipping trajectories."""

    if not rows:
        raise ValueError("G4 trajectory plot requires stage rows")
    colors = {variant: spec.color for variant, spec in G4_SPECS.items()}
    labels = {variant: spec.label for variant, spec in G4_SPECS.items()}
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    for variant in G4_VARIANTS:
        selected = sorted(
            (row for row in rows if row["variant"] == variant),
            key=lambda row: int(row["stage"]),
        )
        if not selected:
            raise ValueError(f"G4 trajectory plot is missing {variant}")
        stages = [int(row["stage"]) for row in selected]
        misfit = [float(row["mean_normalized_data_misfit"]) for row in selected]
        clipping = [
            np.nan
            if row["clipped_fraction"] is None
            else 100.0 * float(row["clipped_fraction"])
            for row in selected
        ]
        spec = G4_SPECS[variant]
        linestyle = "--" if spec.localized else ("-." if spec.bounded else "-")
        axes[0].plot(
            stages,
            misfit,
            marker="o",
            color=colors[variant],
            linestyle=linestyle,
            label=labels[variant],
        )
        axes[1].plot(
            stages,
            clipping,
            marker="o",
            color=colors[variant],
            linestyle=linestyle,
            label=labels[variant],
        )
    axes[0].axhline(1.0, color="0.45", linewidth=1.0, linestyle=":")
    axes[0].set(
        xlabel="ES-MDA stage",
        ylabel="Mean normalized data misfit",
        title="Data fit",
        yscale="log",
    )
    axes[1].set(
        xlabel="ES-MDA stage",
        ylabel="Active cells clipped (%)",
        title="Bound activation",
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.set_xticks(range(9))
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=4)
    figure.suptitle("Egg G4 bound, localization, and ensemble-size controls")
    description = json.dumps({"variants": G4_VARIANTS})
    for path in (svg_path, pdf_path):
        save_vector_figure(
            figure,
            path,
            creator="fmgeo g4_build_controls.py",
            description=description,
        )
    plt.close(figure)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--realizations-dir", type=Path, required=True)
    parser.add_argument("--active-source", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--inversion", action="append", required=True)
    parser.add_argument("--inversion-report", action="append", required=True)
    parser.add_argument("--table-output", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, required=True)
    parser.add_argument("--git-commit")
    args = parser.parse_args()

    inversions = _parse_labeled_paths(args.inversion)
    reports = _parse_labeled_paths(args.inversion_report)
    if tuple(inversions) != G4_VARIANTS or tuple(reports) != G4_VARIANTS:
        raise ValueError(f"G4 inputs must be ordered as: {', '.join(G4_VARIANTS)}")
    stage_rows = _stage_rows(reports)
    config = load_egg_inversion_config(args.config)
    evaluation = _load_report(args.evaluation_report)
    threshold = float(evaluation["high_permeability_threshold_logk"])
    with h5py.File(args.active_source) as handle:
        active = np.asarray(handle["active_mask"], dtype=bool)
    truth = load_egg_ensemble(args.realizations_dir)[
        config.truth_realization - 1 : config.truth_realization
    ]
    wells = parse_egg_well_locations(args.deck.read_text(encoding="utf-8", errors="replace"))
    pair_count = len([name for name in wells if name.startswith("INJECT")]) * len(
        [name for name in wells if name.startswith("PROD")]
    )
    if pair_count != 32 or set(name for name in wells if name.startswith("PROD")) != set(
        EGG_PRODUCERS
    ):
        raise ValueError("official Egg deck must define all 32 injector-producer pairs")
    rows = _build_rows(
        inversions,
        reports=reports,
        truth=truth,
        active=active,
        threshold=threshold,
        wells=wells,
        derived_git_commit=args.git_commit or _git_commit(),
    )
    _write_csv(args.table_output, rows)
    _plot(stage_rows, args.figure_output, args.pdf_output)
    print(
        json.dumps(
            {
                "figure_output": str(args.figure_output),
                "pdf_output": str(args.pdf_output),
                "row_count": len(rows),
                "table_output": str(args.table_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
