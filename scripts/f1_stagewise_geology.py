"""Build the stagewise and matched-misfit Egg geology controls."""

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
from fmgeo.forward.egg import EGG_PRODUCERS, parse_egg_well_locations
from fmgeo.inverse.egg import load_egg_inversion_config
from fmgeo.metrics.egg import egg_well_connectivity, summarize_egg_stage
from fmgeo.plotting import save_vector_figure

METHODS = ("raw", "pca", "fm")
MATCHED_VARIANTS = ("raw_na8", "pca_na8", "fm_na8")
DIAGNOSTIC_SCOPE = (
    "diagnostic stage comparison; an intermediate ES-MDA stage is not equivalent "
    "to a completed run with fewer assimilations"
)


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError("expected a numeric value")
    return float(value)


def _as_int(value: object) -> int:
    if not isinstance(value, (int, np.integer)):
        raise TypeError("expected an integer value")
    return int(value)


def _stage_names(handle: h5py.File) -> list[str]:
    stages = sorted(
        (int(name.removeprefix("stage_")), name)
        for name in handle
        if name.startswith("stage_")
    )
    if not stages:
        raise ValueError("inversion artifact contains no assimilation stages")
    return [name for _, name in stages]


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report root must be an object: {path}")
    return payload


def _build_rows(
    inversions: Mapping[str, Path],
    *,
    reports: Mapping[str, Path],
    truth: np.ndarray,
    active: np.ndarray,
    threshold: float,
    wells: Mapping[str, Sequence[int]],
    derived_git_commit: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Derive stagewise and matched-misfit rows from immutable artifacts."""

    if not inversions or set(inversions) != set(reports):
        raise ValueError("every labeled inversion must have one matching report")
    truth_connectivity = egg_well_connectivity(truth, threshold=threshold, wells=wells)
    stagewise: list[dict[str, object]] = []
    truth_fopt: float | None = None
    strategy: str | None = None
    for variant in inversions:
        artifact_path = inversions[variant]
        report_path = reports[variant]
        report = _load_report(report_path)
        method = str(report["method"])
        if method not in METHODS:
            raise ValueError(f"unsupported inversion method: {method}")
        method_truth_fopt = float(report["truth_terminal_fopt"])
        if truth_fopt is None:
            truth_fopt = method_truth_fopt
        elif not np.isclose(method_truth_fopt, truth_fopt, rtol=0.0, atol=1e-9):
            raise ValueError("inversion reports disagree on truth FOPT")
        artifact_hash = sha256_file(artifact_path)
        report_hash = sha256_file(report_path)
        source_git_commit = str(report["artifact"]["git_commit"])
        with h5py.File(artifact_path) as handle:
            artifact_method = str(handle.attrs["method"])
            artifact_strategy = str(handle.attrs["strategy"])
            parameterization = str(
                handle.attrs.get("parameterization_label", artifact_method)
            )
            if artifact_method != method:
                raise ValueError("artifact/report method does not match its input label")
            if parameterization != variant:
                raise ValueError("artifact parameterization label does not match input label")
            if strategy is None:
                strategy = artifact_strategy
            elif artifact_strategy != strategy:
                raise ValueError("inversion artifacts use different training strategies")
            if str(report["strategy"]) != artifact_strategy:
                raise ValueError("artifact and report strategies do not match")
            observation = np.asarray(handle["observation"], dtype=np.float64)
            sigma = np.asarray(handle["sigma"], dtype=np.float64)
            covariance = np.diag(sigma**2)
            for stage_name in _stage_names(handle):
                group = handle[stage_name]
                summary = summarize_egg_stage(
                    np.asarray(group["logk"], dtype=np.float64),
                    simulated_data=np.asarray(group["simulated_data"], dtype=np.float64),
                    observation=observation,
                    observation_covariance=covariance,
                    fopt=np.asarray(group["fopt"], dtype=np.float64),
                    truth_fopt=truth_fopt,
                    active_mask=active,
                    threshold=threshold,
                    wells=wells,
                    truth_connectivity=truth_connectivity,
                )
                stagewise.append(
                    {
                        "strategy": artifact_strategy,
                        "variant": variant,
                        "parameterization": method,
                        "stage": int(stage_name.removeprefix("stage_")),
                        **summary,
                        "source_artifact": artifact_path.as_posix(),
                        "source_artifact_sha256": artifact_hash,
                        "source_report": report_path.as_posix(),
                        "source_report_sha256": report_hash,
                        "source_git_commit": source_git_commit,
                        "derived_git_commit": derived_git_commit,
                    }
                )
    if truth_fopt is None:
        raise ValueError("truth FOPT is unavailable")

    if set(MATCHED_VARIANTS).issubset(inversions):
        raw_variant, pca_variant, fm_variant = MATCHED_VARIANTS
    elif set(METHODS).issubset(inversions):
        raw_variant, pca_variant, fm_variant = METHODS
    else:
        raise ValueError("matched comparison requires raw, PCA, and FM variants")
    by_variant = {
        variant: [row for row in stagewise if row["variant"] == variant]
        for variant in (raw_variant, pca_variant, fm_variant)
    }
    fm_final = max(by_variant[fm_variant], key=lambda row: _as_int(row["stage"]))
    target = _as_float(fm_final["mean_normalized_data_misfit"])
    matched: list[dict[str, object]] = [
        {
            **fm_final,
            "misfit_gap": 0.0,
            "misfit_relation": "target",
            "matched_target_misfit": target,
            "comparison_scope": DIAGNOSTIC_SCOPE,
        }
    ]
    for variant in (raw_variant, pca_variant):
        rows = by_variant[variant]
        exact = [
            row
            for row in rows
            if np.isclose(
                _as_float(row["mean_normalized_data_misfit"]),
                target,
                rtol=0.0,
                atol=1e-12,
            )
        ]
        if exact:
            selected_rows = [(exact[0], "target")]
        else:
            above = [
                row
                for row in rows
                if _as_float(row["mean_normalized_data_misfit"]) > target
            ]
            below = [
                row
                for row in rows
                if _as_float(row["mean_normalized_data_misfit"]) < target
            ]
            selected_rows = []
            if above:
                selected_rows.append(
                    (
                        min(
                            above,
                            key=lambda row: _as_float(
                                row["mean_normalized_data_misfit"]
                            ),
                        ),
                        "above",
                    )
                )
            if below:
                selected_rows.append(
                    (
                        max(
                            below,
                            key=lambda row: _as_float(
                                row["mean_normalized_data_misfit"]
                            ),
                        ),
                        "below",
                    )
                )
            if not selected_rows:
                raise ValueError(f"variant {variant} has no stages")
        for selected, relation in selected_rows:
            gap = abs(_as_float(selected["mean_normalized_data_misfit"]) - target)
            matched.append(
                {
                    **selected,
                    "misfit_gap": gap,
                    "misfit_relation": relation,
                    "matched_target_misfit": target,
                    "comparison_scope": DIAGNOSTIC_SCOPE,
                }
        )
    return stagewise, matched


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, object]], svg_path: Path, pdf_path: Path) -> None:
    if not rows:
        raise ValueError("trade-off plot requires stagewise rows")
    colors = {"raw": "#440154", "pca": "#21918c", "fm": "#d49b00"}
    linestyles = {
        "raw": "--",
        "pca": "--",
        "fm": "--",
        "raw_na8": "-",
        "pca_na8": "-",
        "fm_na8": "-",
        "fm_na16": ":",
        "fm_na8_n200": "-.",
    }
    labels = {
        "raw": "Raw, Nₐ=4, N=100",
        "pca": "PCA, Nₐ=4, N=100",
        "fm": "FM, Nₐ=4, N=100",
        "raw_na8": "Raw, Nₐ=8, N=100",
        "pca_na8": "PCA, Nₐ=8, N=100",
        "fm_na8": "FM, Nₐ=8, N=100",
        "fm_na16": "FM, Nₐ=16, N=100",
        "fm_na8_n200": "FM, Nₐ=8, N=200",
    }
    annotated_stages = {
        "raw": {1, 2, 4},
        "pca": {1, 2, 4},
        "fm": {4},
        "raw_na8": {1, 2, 8},
        "pca_na8": {1, 2, 8},
        "fm_na8": {8},
        "fm_na16": {16},
        "fm_na8_n200": {8},
    }
    annotation_offsets = {
        "raw_na8": (4, 7),
        "pca_na8": (4, -13),
        "fm_na8": (4, 8),
        "fm_na16": (4, -13),
        "fm_na8_n200": (4, 8),
    }
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    variants = list(dict.fromkeys(str(row["variant"]) for row in rows))
    for variant in variants:
        selected = sorted(
            (row for row in rows if row["variant"] == variant),
            key=lambda row: _as_int(row["stage"]),
        )
        if not selected:
            raise ValueError(f"trade-off plot is missing {variant}")
        method = str(selected[0]["parameterization"])
        misfit = np.asarray(
            [_as_float(row["mean_normalized_data_misfit"]) for row in selected]
        )
        panels = (
            ("bimodality_coefficient", "Bimodality coefficient"),
            ("connectivity_mae_to_truth", "Connectivity MAE to truth"),
        )
        for axis, (metric, label) in zip(axes, panels, strict=True):
            values = np.asarray([_as_float(row[metric]) for row in selected])
            axis.plot(
                misfit,
                values,
                marker="o",
                color=colors[method],
                linestyle=linestyles.get(variant, "-"),
                label=labels.get(variant, variant),
            )
            for row, x, y in zip(selected, misfit, values, strict=True):
                stage = _as_int(row["stage"])
                if stage not in annotated_stages.get(variant, {0, len(selected) - 1}):
                    continue
                axis.annotate(
                    str(stage),
                    (x, y),
                    xytext=annotation_offsets.get(variant, (4, 6)),
                    textcoords="offset points",
                    fontsize=8,
                )
            axis.set(xlabel="Mean normalized data misfit", ylabel=label)
            axis.set_xscale("log")
            axis.margins(x=0.08, y=0.1)
            axis.grid(alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="outside lower center", ncol=3)
    figure.suptitle("Egg assimilation trajectory: data fit versus geology")
    description = json.dumps(
        sorted({str(row["source_artifact_sha256"]) for row in rows})
    )
    for path in (svg_path, pdf_path):
        save_vector_figure(
            figure,
            path,
            creator="fmgeo f1_stagewise_geology.py",
            description=description,
        )
    plt.close(figure)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _map_inversions(paths: list[Path]) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    for path in paths:
        with h5py.File(path) as handle:
            method = str(handle.attrs["method"])
            label = str(handle.attrs.get("parameterization_label", method))
        if label in mapped:
            raise ValueError(f"duplicate inversion label: {label}")
        mapped[label] = path
    return mapped


def _map_reports(paths: list[Path]) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    for path in paths:
        report = _load_report(path)
        method = str(report["method"])
        label = str(report.get("parameterization_label", method))
        if label in mapped:
            raise ValueError(f"duplicate inversion report label: {label}")
        mapped[label] = path
    return mapped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--realizations-dir", type=Path, required=True)
    parser.add_argument("--active-source", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--inversion", type=Path, action="append", required=True)
    parser.add_argument("--inversion-report", type=Path, action="append", required=True)
    parser.add_argument("--stagewise-output", type=Path, required=True)
    parser.add_argument("--matched-output", type=Path, required=True)
    parser.add_argument("--figure-output", type=Path, required=True)
    parser.add_argument("--pdf-output", type=Path, required=True)
    parser.add_argument("--git-commit")
    args = parser.parse_args()

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
    stagewise, matched = _build_rows(
        _map_inversions(args.inversion),
        reports=_map_reports(args.inversion_report),
        truth=truth,
        active=active,
        threshold=threshold,
        wells=wells,
        derived_git_commit=args.git_commit or _git_commit(),
    )
    _write_csv(args.stagewise_output, stagewise)
    _write_csv(args.matched_output, matched)
    _plot(stagewise, args.figure_output, args.pdf_output)
    print(
        json.dumps(
            {
                "stagewise_rows": len(stagewise),
                "matched_rows": len(matched),
                "stagewise_output": str(args.stagewise_output),
                "matched_output": str(args.matched_output),
                "figure_output": str(args.figure_output),
                "pdf_output": str(args.pdf_output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
