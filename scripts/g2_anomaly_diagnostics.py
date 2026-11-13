"""Diagnose non-monotone FM remedy runs from immutable inversion artifacts."""

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

from fmgeo.artifacts import sha256_file
from fmgeo.inverse.egg import load_egg_inversion_config

REQUIRED_VARIANTS = ("fm", "fm_na8", "fm_na16", "fm_na8_n200")
CORE_REPORT_FIELDS = (
    "prior_sha256",
    "truth_case",
    "observation_count",
    "observation_noise_seed",
)
CORE_PARAMETERIZATION_FIELDS = (
    "checkpoint_sha256",
    "fm_config_hash",
    "training_data_sha256",
)


def _load_report(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"report root must be an object: {path}")
    return payload


def _stage_names(handle: h5py.File) -> list[str]:
    names = sorted(
        (int(name.removeprefix("stage_")), name)
        for name in handle
        if name.startswith("stage_")
    )
    if not names:
        raise ValueError("inversion artifact contains no stages")
    return [name for _, name in names]


def _relative_shift(reference: np.ndarray, current: np.ndarray) -> float:
    denominator = float(np.linalg.norm(reference.ravel()))
    numerator = float(np.linalg.norm((current - reference).ravel()))
    return numerator / denominator if denominator > 0.0 else float("nan")


def _effective_ensemble_members(fields: np.ndarray) -> float:
    flattened = fields.reshape(len(fields), -1)
    anomalies = flattened - flattened.mean(axis=0)
    eigenvalues = np.linalg.eigvalsh(anomalies @ anomalies.T)
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    energy = float(eigenvalues.sum())
    participation = (
        0.0 if energy == 0.0 else energy**2 / float(np.sum(eigenvalues**2))
    )
    return min(float(len(fields)), 1.0 + participation)


def _validate_common_provenance(reports: Mapping[str, Mapping[str, Any]]) -> None:
    reference = reports["fm_na8"]
    for field in CORE_REPORT_FIELDS:
        expected = reference[field]
        for variant, report in reports.items():
            if report[field] != expected:
                raise ValueError(f"{field} differs for {variant}")
    reference_parameterization = reference["parameterization"]
    for field in CORE_PARAMETERIZATION_FIELDS:
        expected = reference_parameterization[field]
        for variant, report in reports.items():
            if report["parameterization"][field] != expected:
                raise ValueError(f"parameterization/{field} differs for {variant}")


def _build_rows(
    inversions: Mapping[str, Path],
    *,
    reports: Mapping[str, Path],
    inflations: Mapping[str, Sequence[float]],
    derived_git_commit: str,
    config_paths: Mapping[str, Path] | None = None,
) -> list[dict[str, object]]:
    """Measure stagewise saturation, conditioning, and ensemble information."""

    expected = set(REQUIRED_VARIANTS)
    if set(inversions) != expected or set(reports) != expected or set(inflations) != expected:
        raise ValueError(f"inputs must contain exactly {sorted(expected)}")
    loaded_reports = {variant: _load_report(path) for variant, path in reports.items()}
    _validate_common_provenance(loaded_reports)
    paths = {} if config_paths is None else dict(config_paths)

    observations: dict[str, np.ndarray] = {}
    sigmas: dict[str, np.ndarray] = {}
    stage0_fields: dict[str, np.ndarray] = {}
    artifact_hashes: dict[str, str] = {}
    for variant in REQUIRED_VARIANTS:
        path = inversions[variant]
        report = loaded_reports[variant]
        artifact_hash = sha256_file(path)
        if report["artifact"]["sha256"] != artifact_hash:
            raise ValueError(f"artifact hash differs from report for {variant}")
        artifact_hashes[variant] = artifact_hash
        values = tuple(float(value) for value in inflations[variant])
        reciprocal_sum = float(np.sum(1.0 / np.asarray(values)))
        if not np.isclose(reciprocal_sum, 1.0, rtol=0.0, atol=1e-12):
            raise ValueError(f"inflations are invalid for {variant}")
        if int(report.get("n_assimilations", len(values))) != len(values):
            raise ValueError(f"assimilation count differs for {variant}")
        with h5py.File(path) as handle:
            method = str(handle.attrs["method"])
            label = str(handle.attrs.get("parameterization_label", method))
            if method != "fm" or label != variant:
                raise ValueError(f"artifact label differs for {variant}")
            names = _stage_names(handle)
            if len(names) != len(values) + 1:
                raise ValueError(f"stage count differs for {variant}")
            observations[variant] = np.asarray(handle["observation"], dtype=np.float64)
            sigmas[variant] = np.asarray(handle["sigma"], dtype=np.float64)
            stage0 = handle[names[0]]
            parameters = np.asarray(stage0["parameters"], dtype=np.float64)
            fields = np.asarray(stage0["logk"], dtype=np.float64)
            if len(parameters) != len(fields) or not np.isfinite(parameters).all():
                raise ValueError(f"invalid stage-0 inverse transform for {variant}")
            if not np.isfinite(fields).all():
                raise ValueError(f"invalid stage-0 fields for {variant}")
            stage0_fields[variant] = fields

    reference_observation = observations["fm_na8"]
    reference_sigma = sigmas["fm_na8"]
    for variant in REQUIRED_VARIANTS:
        if not np.array_equal(observations[variant], reference_observation):
            raise ValueError(f"observation differs for {variant}")
        if not np.array_equal(sigmas[variant], reference_sigma):
            raise ValueError(f"sigma differs for {variant}")

    reference_fields = stage0_fields["fm_na8"]
    rows: list[dict[str, object]] = []
    for variant in REQUIRED_VARIANTS:
        path = inversions[variant]
        report_path = reports[variant]
        report = loaded_reports[variant]
        values = tuple(float(value) for value in inflations[variant])
        reciprocal_sum = float(np.sum(1.0 / np.asarray(values)))
        candidate_prefix = stage0_fields[variant][: len(reference_fields)]
        prefix_relative_fro = _relative_shift(reference_fields, candidate_prefix)
        prefix_matches = bool(
            np.allclose(reference_fields, candidate_prefix, rtol=1e-6, atol=5e-5)
        )
        config_path = paths.get(variant)
        config_hash = sha256_file(config_path) if config_path is not None else ""
        with h5py.File(path) as handle:
            names = _stage_names(handle)
            initial_fields = np.asarray(handle[names[0]]["logk"], dtype=np.float64)
            previous_fields = initial_fields
            observation = np.asarray(handle["observation"], dtype=np.float64)
            sigma = np.asarray(handle["sigma"], dtype=np.float64)
            for stage_index, stage_name in enumerate(names):
                group = handle[stage_name]
                fields = np.asarray(group["logk"], dtype=np.float64)
                simulated = np.asarray(group["simulated_data"], dtype=np.float64)
                normalized = simulated / sigma[None, :]
                covariance = np.atleast_2d(np.cov(normalized, rowvar=False, ddof=1))
                alpha = values[stage_index] if stage_index < len(values) else values[-1]
                system = covariance + alpha * np.eye(len(sigma))
                ensemble_variance = float(np.trace(covariance))
                total_variance = ensemble_variance + alpha * len(sigma)
                data_variance_fraction = ensemble_variance / total_variance
                normalized_residual = (simulated - observation[None, :]) / sigma[None, :]
                effective_members = (
                    float(group.attrs["effective_ensemble_members"])
                    if "effective_ensemble_members" in group.attrs
                    else _effective_ensemble_members(fields)
                )
                step_shift = (
                    0.0
                    if stage_index == 0
                    else _relative_shift(previous_fields, fields)
                )
                rows.append(
                    {
                        "strategy": report["strategy"],
                        "variant": variant,
                        "parameterization": "fm",
                        "stage": stage_index,
                        "n_assimilations": len(values),
                        "ensemble_size": len(fields),
                        "alpha": alpha,
                        "inflation_reciprocal_sum": reciprocal_sum,
                        "mean_normalized_data_misfit": float(
                            np.mean(normalized_residual**2)
                        ),
                        "field_step_relative_shift": step_shift,
                        "distance_from_stage0": _relative_shift(initial_fields, fields),
                        "condition_number": float(np.linalg.cond(system)),
                        "ensemble_data_variance_fraction": data_variance_fraction,
                        "effective_ensemble_members": effective_members,
                        "stage0_parameter_count": int(handle[names[0]]["parameters"].shape[0]),
                        "stage0_prefix_matches_fm_na8": prefix_matches,
                        "stage0_prefix_relative_fro": prefix_relative_fro,
                        "configuration_consistent": True,
                        "parameterization_device": report["parameterization"].get(
                            "device", ""
                        ),
                        "source_artifact": path.as_posix(),
                        "source_artifact_sha256": artifact_hashes[variant],
                        "source_report": report_path.as_posix(),
                        "source_report_sha256": sha256_file(report_path),
                        "protocol_config": config_path.as_posix() if config_path else "",
                        "protocol_config_sha256": config_hash,
                        "source_git_commit": report["artifact"]["git_commit"],
                        "derived_git_commit": derived_git_commit,
                    }
                )
                previous_fields = fields
    return rows


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
            raise ValueError(f"duplicate report label: {label}")
        mapped[label] = path
    return mapped


def _parse_config_maps(values: list[str]) -> tuple[dict[str, tuple[float, ...]], dict[str, Path]]:
    inflations: dict[str, tuple[float, ...]] = {}
    paths: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("config maps must use LABEL=PATH")
        path = Path(raw_path)
        config = load_egg_inversion_config(path)
        inflations[label] = tuple(config.inflations)
        paths[label] = path
    return inflations, paths


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty anomaly diagnostics")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inversion", type=Path, action="append", required=True)
    parser.add_argument("--inversion-report", type=Path, action="append", required=True)
    parser.add_argument("--config-map", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    inflations, config_paths = _parse_config_maps(args.config_map)
    rows = _build_rows(
        _map_inversions(args.inversion),
        reports=_map_reports(args.inversion_report),
        inflations=inflations,
        config_paths=config_paths,
        derived_git_commit=args.git_commit or _git_commit(),
    )
    _write_csv(args.output, rows)
    print(json.dumps({"output": str(args.output), "row_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
