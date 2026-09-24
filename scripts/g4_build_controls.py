"""Build the final bounded G4 Egg comparison table and trajectory figure."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import matplotlib.pyplot as plt
import numpy as np
from f2_build_remedies import _build_rows, _load_report, _write_csv
from f3_build_inversion_v2 import _parse_labeled_paths
from m8_egg_case import load_egg_ensemble

from fmgeo.forward.egg import EGG_PRODUCERS, parse_egg_well_locations
from fmgeo.inverse.egg import load_egg_inversion_config
from fmgeo.plotting import save_vector_figure

G4_VARIANTS = (
    "raw_na8",
    "pca_na8",
    "fm_na8",
    "raw_na8_localized",
    "fm_na8_localized",
    "raw_na8_n200",
)


def _stage_rows(reports: Mapping[str, Path]) -> list[dict[str, object]]:
    """Load stage trajectories in the fixed G4 comparison order."""

    if tuple(reports) != G4_VARIANTS:
        raise ValueError(f"G4 reports must be ordered as: {', '.join(G4_VARIANTS)}")
    rows: list[dict[str, object]] = []
    for variant, path in reports.items():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("parameterization_label") != variant:
            raise ValueError(f"G4 report label does not match input variant: {variant}")
        for stage in report["stages"]:
            rows.append(
                {
                    "variant": variant,
                    "stage": int(stage["stage"]),
                    "mean_normalized_data_misfit": float(stage["mean_normalized_data_misfit"]),
                    "clipped_fraction": float(stage.get("clipped_fraction", 0.0)),
                }
            )
    return rows


def _plot(rows: list[dict[str, object]], svg_path: Path, pdf_path: Path) -> None:
    """Plot G4 data-fit and clipping trajectories."""

    if not rows:
        raise ValueError("G4 trajectory plot requires stage rows")
    colors = {
        "raw_na8": "#440154",
        "pca_na8": "#21918c",
        "fm_na8": "#d49b00",
        "raw_na8_localized": "#7e1e9c",
        "fm_na8_localized": "#d95f02",
        "raw_na8_n200": "#31688e",
    }
    labels = {
        "raw_na8": "Raw, N=100",
        "pca_na8": "PCA, N=100",
        "fm_na8": "FM, N=100",
        "raw_na8_localized": "Raw localized, N=100",
        "fm_na8_localized": "FM localized, N=100",
        "raw_na8_n200": "Raw, N=200",
    }
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
        clipping = [100.0 * float(row["clipped_fraction"]) for row in selected]
        linestyle = "--" if "localized" in variant else "-"
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
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=3)
    figure.suptitle("Egg G4 controls: localization destabilizes bounded updates")
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
