"""Build the provenance-complete Egg inversion-remedy comparison table."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
from m8_egg_case import load_egg_ensemble

from fmgeo.artifacts import sha256_file
from fmgeo.forward.egg import EGG_PRODUCERS, parse_egg_well_locations
from fmgeo.inverse.egg import load_egg_inversion_config
from fmgeo.metrics.egg import egg_well_connectivity, summarize_egg_stage
from fmgeo.metrics.uq import crps_ensemble, energy_score, interval_width

METHODS = ("raw", "pca", "fm")


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report root must be an object: {path}")
    return payload


def _last_stage(handle: h5py.File) -> str:
    stages = sorted(
        (int(name.removeprefix("stage_")), name)
        for name in handle
        if name.startswith("stage_")
    )
    if not stages:
        raise ValueError("inversion artifact contains no stages")
    return stages[-1][1]


def _build_rows(
    inversions: Mapping[str, Path],
    *,
    reports: Mapping[str, Path],
    truth: np.ndarray,
    active: np.ndarray,
    threshold: float,
    wells: Mapping[str, Sequence[int]],
    derived_git_commit: str,
) -> list[dict[str, object]]:
    """Summarize final stages with geology, UQ, collapse, and provenance."""

    if set(inversions) != set(METHODS) or set(reports) != set(METHODS):
        raise ValueError("exactly one raw, PCA, and FM artifact and report are required")
    truth_connectivity = egg_well_connectivity(truth, threshold=threshold, wells=wells)
    rows: list[dict[str, object]] = []
    for method in METHODS:
        artifact_path = inversions[method]
        report_path = reports[method]
        report = _load_report(report_path)
        artifact_hash = sha256_file(artifact_path)
        if str(report["method"]) != method:
            raise ValueError("remedy report method does not match its input label")
        if str(report["artifact"]["sha256"]) != artifact_hash:
            raise ValueError("remedy artifact hash does not match its report")
        final_report = report["stages"][-1]
        with h5py.File(artifact_path) as handle:
            if str(handle.attrs["method"]) != method:
                raise ValueError("remedy artifact method does not match its input label")
            parameterization_label = str(
                handle.attrs.get("parameterization_label", method)
            )
            stage_name = _last_stage(handle)
            group = handle[stage_name]
            fields = np.asarray(group["logk"], dtype=np.float64)
            simulated = np.asarray(group["simulated_data"], dtype=np.float64)
            fopt = np.asarray(group["fopt"], dtype=np.float64)
            observation = np.asarray(handle["observation"], dtype=np.float64)
            truth_data = np.asarray(handle["truth_data"], dtype=np.float64)
            sigma = np.asarray(handle["sigma"], dtype=np.float64)
            summary = summarize_egg_stage(
                fields,
                simulated_data=simulated,
                observation=observation,
                observation_covariance=np.diag(sigma**2),
                fopt=fopt,
                truth_fopt=float(report["truth_terminal_fopt"]),
                active_mask=active,
                threshold=threshold,
                wells=wells,
                truth_connectivity=truth_connectivity,
            )
        rows.append(
            {
                "strategy": report["strategy"],
                "parameterization": method,
                "parameterization_label": parameterization_label,
                "remedy": report["remedy"],
                "n_assimilations": report["n_assimilations"],
                "localization_enabled": report["localization_enabled"],
                "localization_radius_m": report["localization_radius_m"],
                "latent_rank": report["latent_rank"],
                "ensemble_size": len(fields),
                **summary,
                "P10_P90_width": interval_width(
                    float(summary["fopt_p10"]), float(summary["fopt_p90"])
                ),
                "fopt_crps": crps_ensemble(
                    fopt, observation=float(report["truth_terminal_fopt"])
                ),
                "normalized_observation_energy_score": energy_score(
                    simulated / sigma[None, :], observation=truth_data / sigma
                ),
                "mean_field_spread": final_report["mean_field_spread"],
                "relative_field_shift_from_prior": final_report[
                    "relative_field_shift_from_prior"
                ],
                "effective_ensemble_members": final_report[
                    "effective_ensemble_members"
                ],
                "n_sim": report["n_sim"],
                "n_failed": report["n_failed"],
                "source_report": report_path.as_posix(),
                "source_report_sha256": sha256_file(report_path),
                "config_hash": report["config_hash"],
                "source_artifact": report["artifact"]["path"],
                "source_artifact_sha256": artifact_hash,
                "source_git_commit": report["artifact"]["git_commit"],
                "derived_git_commit": derived_git_commit,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty remedy table")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _map_inversions(paths: list[Path]) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    for path in paths:
        with h5py.File(path) as handle:
            method = str(handle.attrs["method"])
        if method in mapped:
            raise ValueError(f"duplicate inversion method: {method}")
        mapped[method] = path
    return mapped


def _map_reports(paths: list[Path]) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    for path in paths:
        method = str(_load_report(path)["method"])
        if method in mapped:
            raise ValueError(f"duplicate report method: {method}")
        mapped[method] = path
    return mapped


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
    parser.add_argument("--inversion", type=Path, action="append", required=True)
    parser.add_argument("--inversion-report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
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
    wells = parse_egg_well_locations(
        args.deck.read_text(encoding="utf-8", errors="replace")
    )
    pair_count = len([name for name in wells if name.startswith("INJECT")]) * len(
        [name for name in wells if name.startswith("PROD")]
    )
    if pair_count != 32 or set(name for name in wells if name.startswith("PROD")) != set(
        EGG_PRODUCERS
    ):
        raise ValueError("official Egg deck must define all 32 injector-producer pairs")
    rows = _build_rows(
        _map_inversions(args.inversion),
        reports=_map_reports(args.inversion_report),
        truth=truth,
        active=active,
        threshold=threshold,
        wells=wells,
        derived_git_commit=args.git_commit or _git_commit(),
    )
    _write_csv(args.output, rows)
    print(json.dumps({"output": str(args.output), "row_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
