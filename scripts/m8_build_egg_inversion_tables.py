"""Derive Egg connectivity, bimodality, and breakthrough tables from raw artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
from m8_egg_case import load_egg_ensemble

from fmgeo.artifacts import sha256_file
from fmgeo.forward.egg import (
    EGG_PRODUCERS,
    extract_egg_observations,
    parse_egg_well_locations,
)
from fmgeo.inverse.egg import load_egg_inversion_config
from fmgeo.metrics.egg import bimodality_coefficient, egg_well_connectivity


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _last_stage(handle: h5py.File) -> str:
    stages = sorted(
        (int(name.removeprefix("stage_")), name)
        for name in handle
        if name.startswith("stage_")
    )
    if not stages:
        raise ValueError("inversion artifact contains no stages")
    return stages[-1][1]


def _metadata(path: Path, stage: str) -> list[dict[str, Any]]:
    with h5py.File(path) as handle:
        raw = np.asarray(handle[stage]["metadata_json"])
    return [json.loads(item.decode() if isinstance(item, bytes) else str(item)) for item in raw]


def _breakthrough_row(
    *,
    strategy: str,
    category: str,
    producer: str,
    metadata: list[dict[str, Any]],
    truth_day: float | None,
    source: Path,
    source_hash: str,
    derived_git_commit: str,
) -> dict[str, object]:
    values = [item["water_breakthrough_day"][producer] for item in metadata]
    finite = np.asarray([float(value) for value in values if value is not None])
    quantiles = [float("nan")] * 3 if len(finite) == 0 else np.quantile(finite, [0.1, 0.5, 0.9])
    return {
        "strategy": strategy,
        "category": category,
        "producer": producer,
        "P10_day_conditional": quantiles[0],
        "P50_day_conditional": quantiles[1],
        "P90_day_conditional": quantiles[2],
        "censored_fraction": 1.0 - len(finite) / len(values),
        "truth_day": truth_day,
        "source_artifact": source.as_posix(),
        "source_artifact_sha256": source_hash,
        "derived_git_commit": derived_git_commit,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--truth-case", type=Path, required=True)
    parser.add_argument("--realizations-dir", type=Path, required=True)
    parser.add_argument("--active-source", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--inversion", type=Path, action="append", required=True)
    parser.add_argument("--inversion-report", type=Path, action="append", required=True)
    parser.add_argument("--connectivity-output", type=Path, required=True)
    parser.add_argument("--bimodality-output", type=Path, required=True)
    parser.add_argument("--breakthrough-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    args = parser.parse_args()

    config = load_egg_inversion_config(args.config)
    derived_git_commit = _git_commit()
    evaluation = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
    threshold = float(evaluation["high_permeability_threshold_logk"])
    strategy = str(evaluation["strategy"])
    wells = parse_egg_well_locations(args.deck.read_text(encoding="utf-8", errors="replace"))
    if len([name for name in wells if name.startswith("INJECT")]) != 8 or set(
        name for name in wells if name.startswith("PROD")
    ) != set(EGG_PRODUCERS):
        raise ValueError("official Egg deck must define 8 injectors and 4 producers")
    with h5py.File(args.active_source) as handle:
        active = np.asarray(handle["active_mask"], dtype=bool)
    reference = load_egg_ensemble(args.realizations_dir)
    truth = reference[config.truth_realization - 1 : config.truth_realization]
    truth_source = args.realizations_dir / f"PERM{config.truth_realization}_ECL.INC"

    artifacts: dict[str, tuple[Path, str, np.ndarray, list[dict[str, Any]]]] = {}
    for path in args.inversion:
        digest = sha256_file(path)
        with h5py.File(path) as handle:
            method = str(handle.attrs["method"])
            artifact_strategy = str(handle.attrs["strategy"])
            if artifact_strategy != strategy:
                raise ValueError("inversion and evaluation strategies do not match")
            final_stage = _last_stage(handle)
            fields = np.asarray(handle[final_stage]["logk"], dtype=np.float64)
            if method == "raw":
                prior = np.asarray(handle["stage_0"]["logk"], dtype=np.float64)
        artifacts[method] = (path, digest, fields, _metadata(path, final_stage))
        if method == "raw":
            artifacts["prior"] = (path, digest, prior, _metadata(path, "stage_0"))
    if set(artifacts) != {"prior", "raw", "pca", "fm"}:
        raise ValueError("tables require one complete prior/raw/PCA/FM comparison")

    connectivity_rows: list[dict[str, object]] = []
    bimodality_rows: list[dict[str, object]] = []
    for category in ("prior", "raw", "pca", "fm"):
        path, digest, fields, _ = artifacts[category]
        connections = egg_well_connectivity(fields, threshold=threshold, wells=wells)
        for (injector, producer), probability in connections.items():
            connectivity_rows.append(
                {
                    "strategy": strategy,
                    "category": category,
                    "injector": injector,
                    "producer": producer,
                    "connectivity_probability": probability,
                    "threshold_logk": threshold,
                    "source_artifact": path.as_posix(),
                    "source_artifact_sha256": digest,
                    "derived_git_commit": derived_git_commit,
                }
            )
        bimodality_rows.append(
            {
                "strategy": strategy,
                "category": category,
                "bimodality_coefficient": bimodality_coefficient(fields[:, active]),
                "threshold_logk": threshold,
                "source_artifact": path.as_posix(),
                "source_artifact_sha256": digest,
                "derived_git_commit": derived_git_commit,
            }
        )
    truth_connections = egg_well_connectivity(truth, threshold=threshold, wells=wells)
    truth_hash = sha256_file(truth_source)
    for (injector, producer), probability in truth_connections.items():
        connectivity_rows.append(
            {
                "strategy": strategy,
                "category": "truth",
                "injector": injector,
                "producer": producer,
                "connectivity_probability": probability,
                "threshold_logk": threshold,
                "source_artifact": truth_source.as_posix(),
                "source_artifact_sha256": truth_hash,
                "derived_git_commit": derived_git_commit,
            }
        )
    bimodality_rows.append(
        {
            "strategy": strategy,
            "category": "truth",
            "bimodality_coefficient": bimodality_coefficient(truth[:, active]),
            "threshold_logk": threshold,
            "source_artifact": truth_source.as_posix(),
            "source_artifact_sha256": truth_hash,
            "derived_git_commit": derived_git_commit,
        }
    )

    truth_summary = extract_egg_observations(
        args.truth_case,
        history_end_day=config.history_end_day,
        observation_interval_days=config.observation_interval_days,
        oil_rate_relative_sigma=config.oil_rate_relative_sigma,
        water_rate_relative_sigma=config.water_rate_relative_sigma,
        rate_sigma_floor=config.rate_sigma_floor,
        water_breakthrough_fraction=config.water_breakthrough_fraction,
    )
    truth_breakthrough = truth_summary["metadata"]["water_breakthrough_day"]
    breakthrough_rows = [
        _breakthrough_row(
            strategy=strategy,
            category=category,
            producer=producer,
            metadata=artifacts[category][3],
            truth_day=truth_breakthrough[producer],
            source=artifacts[category][0],
            source_hash=artifacts[category][1],
            derived_git_commit=derived_git_commit,
        )
        for category in ("prior", "raw", "pca", "fm")
        for producer in EGG_PRODUCERS
    ]
    reports: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in args.inversion_report:
        report = json.loads(path.read_text(encoding="utf-8"))
        if str(report["strategy"]) != strategy:
            raise ValueError("inversion report and evaluation strategies do not match")
        reports[str(report["method"])] = (path, report)
    if set(reports) != {"raw", "pca", "fm"}:
        raise ValueError("summary requires one raw, PCA, and FM report")
    summary_rows: list[dict[str, object]] = []
    for category in ("prior", "raw", "pca", "fm"):
        method = "raw" if category == "prior" else category
        path, report = reports[method]
        stage = report["stages"][0 if category == "prior" else -1]
        fopt = stage["fopt"]
        summary_rows.append(
            {
                "strategy": strategy,
                "category": category,
                "P10": fopt["P10"],
                "P50": fopt["P50"],
                "P90": fopt["P90"],
                "truth": fopt["truth"],
                "covered": fopt["covered"],
                "mean_normalized_data_misfit": stage["mean_normalized_data_misfit"],
                "n_sim": report["n_sim"],
                "n_failed": report["n_failed"],
                "source_report": path.as_posix(),
                "source_report_sha256": sha256_file(path),
                "config_hash": report["config_hash"],
                "source_artifact": report["artifact"]["path"],
                "source_artifact_sha256": report["artifact"]["sha256"],
                "source_git_commit": report["artifact"]["git_commit"],
                "derived_git_commit": derived_git_commit,
            }
        )
    _write_csv(args.connectivity_output, connectivity_rows)
    _write_csv(args.bimodality_output, bimodality_rows)
    _write_csv(args.breakthrough_output, breakthrough_rows)
    _write_csv(args.summary_output, summary_rows)
    print(
        json.dumps(
            {
                "connectivity_rows": len(connectivity_rows),
                "bimodality_rows": len(bimodality_rows),
                "breakthrough_rows": len(breakthrough_rows),
                "summary_rows": len(summary_rows),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
