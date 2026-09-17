"""Measure linear response, decoder nonlinearity, and FM update attenuation."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import h5py  # type: ignore[import-untyped]
import numpy as np
from egg_flow_adapter import build_flow_adapter, embed_active

from fmgeo.artifacts import sha256_file
from fmgeo.inverse.diagnostics import (
    midpoint_nonlinearity,
    relative_frobenius_shift,
    sample_member_pairs,
)
from fmgeo.inverse.egg import load_egg_inversion_config
from fmgeo.inverse.esmda import linear_response_diagnostic
from fmgeo.param.pca import PCAParameterization

METHODS = ("raw", "pca", "fm")
Decoder = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class CrossValidatedLinearResponse:
    """Out-of-sample linear-response scores over deterministic folds."""

    in_sample_r2: float
    mean: float
    std: float
    fold_scores: tuple[float, ...]


def _retained_rank(singular_values: np.ndarray, energy: float) -> int:
    if not 0.0 < energy <= 1.0:
        raise ValueError("svd_energy must be in (0, 1]")
    cutoff = (
        np.finfo(np.float64).eps
        * max(len(singular_values), 1)
        * singular_values[0]
    )
    available = singular_values[singular_values > cutoff]
    if len(available) == 0:
        raise ValueError("training parameter anomalies are numerically singular")
    cumulative = np.cumsum(available**2) / np.sum(available**2)
    return min(int(np.searchsorted(cumulative, energy, side="left")) + 1, len(available))


def _cross_validated_linear_response(
    parameters: np.ndarray,
    simulated_data: np.ndarray,
    *,
    svd_energy: float,
    n_folds: int,
    seed: int,
) -> CrossValidatedLinearResponse:
    """Fit truncated linear maps on training folds and score held-out members."""

    model = np.asarray(parameters, dtype=np.float64)
    data = np.asarray(simulated_data, dtype=np.float64)
    if model.ndim != 2 or data.ndim != 2 or len(model) != len(data):
        raise ValueError("parameters and simulated_data must be ensemble-first matrices")
    if not 2 <= n_folds <= len(model):
        raise ValueError("n_folds must be between two and the ensemble size")
    if not np.isfinite(model).all() or not np.isfinite(data).all():
        raise ValueError("linearity inputs must be finite")
    permutation = np.random.default_rng(seed).permutation(len(model))
    folds = np.array_split(permutation, n_folds)
    scores: list[float] = []
    all_indices = np.arange(len(model))
    for held_out in folds:
        training = np.setdiff1d(all_indices, held_out, assume_unique=True)
        model_mean = model[training].mean(axis=0)
        data_mean = data[training].mean(axis=0)
        model_anomalies = model[training] - model_mean
        data_anomalies = data[training] - data_mean
        left, singular_values, right = np.linalg.svd(
            model_anomalies, full_matrices=False
        )
        rank = _retained_rank(singular_values, svd_energy)
        coefficients = (
            right[:rank].T
            @ ((left[:, :rank].T @ data_anomalies) / singular_values[:rank, None])
        )
        predicted = (model[held_out] - model_mean) @ coefficients + data_mean
        residual_energy = float(np.sum((data[held_out] - predicted) ** 2))
        baseline_energy = float(np.sum((data[held_out] - data_mean) ** 2))
        if baseline_energy <= 0.0 or not np.isfinite(baseline_energy):
            raise ValueError("held-out simulated data must have positive variance")
        scores.append(1.0 - residual_energy / baseline_energy)
    in_sample = linear_response_diagnostic(
        model, data, svd_energy=svd_energy
    ).r2
    return CrossValidatedLinearResponse(
        in_sample_r2=in_sample,
        mean=float(np.mean(scores)),
        std=float(np.std(scores, ddof=1)),
        fold_scores=tuple(float(score) for score in scores),
    )


def _stage_names(handle: h5py.File) -> list[str]:
    stages = sorted(
        (int(name.removeprefix("stage_")), name)
        for name in handle
        if name.startswith("stage_")
    )
    if not stages:
        raise ValueError("inversion artifact contains no stages")
    return [name for _, name in stages]


def _build_diagnostics(
    inversions: Mapping[str, Path],
    *,
    decoders: Mapping[str, Decoder],
    svd_energy: float,
    pair_count: int,
    pair_seed: int,
    n_folds: int,
    fold_seed: int,
    derived_git_commit: str,
    decoder_metadata: Mapping[str, Mapping[str, object]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    """Build one stable diagnostic schema from existing inversion stages."""

    if set(inversions) != set(METHODS) or set(decoders) != set(METHODS):
        raise ValueError("exactly one raw, PCA, and FM inversion and decoder are required")
    metadata = {} if decoder_metadata is None else decoder_metadata
    ensemble_size: int | None = None
    sources: dict[str, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    payload_parameterizations: dict[str, dict[str, object]] = {}
    pairs: np.ndarray | None = None
    strategy: str | None = None
    for method in METHODS:
        path = inversions[method]
        source_hash = sha256_file(path)
        with h5py.File(path) as handle:
            artifact_method = str(handle.attrs["method"])
            artifact_strategy = str(handle.attrs["strategy"])
            label = str(handle.attrs.get("parameterization_label", artifact_method))
            if artifact_method != method or label != method:
                raise ValueError("diagnostic inputs must use raw, pca, and fm labels")
            if strategy is None:
                strategy = artifact_strategy
            elif artifact_strategy != strategy:
                raise ValueError("diagnostic artifacts use different strategies")
            group = handle["stage_0"]
            parameters = np.asarray(group["parameters"], dtype=np.float64)
            simulated = np.asarray(group["simulated_data"], dtype=np.float64)
            decoded = np.asarray(group["logk"], dtype=np.float64)
            if ensemble_size is None:
                ensemble_size = len(parameters)
                pairs = sample_member_pairs(
                    ensemble_size,
                    pair_count,
                    rng=np.random.default_rng(pair_seed),
                )
            elif len(parameters) != ensemble_size:
                raise ValueError("diagnostic ensembles must have the same member count")
            if pairs is None:
                raise AssertionError("pair sampling was not initialized")
            response = linear_response_diagnostic(
                parameters,
                simulated,
                svd_energy=svd_energy,
            )
            cross_validated = _cross_validated_linear_response(
                parameters,
                simulated,
                svd_energy=svd_energy,
                n_folds=n_folds,
                seed=fold_seed,
            )
            midpoint_errors = midpoint_nonlinearity(
                parameters,
                decode=decoders[method],
                pairs=pairs,
                decoded_parameters=decoded,
            )
            p10, p50, p90 = np.quantile(midpoint_errors, (0.1, 0.5, 0.9))
            method_metadata = dict(metadata.get(method, {}))
            checkpoint_hash = str(method_metadata.get("checkpoint_sha256", ""))
            row = {
                "strategy": artifact_strategy,
                "parameterization": method,
                "diagnostic": "stage_0_linearity_and_decoder",
                "stage": 0,
                "r2_lin": response.r2,
                "r2_oos_mean": cross_validated.mean,
                "r2_oos_std": cross_validated.std,
                "n_folds": n_folds,
                "fold_seed": fold_seed,
                "retained_rank": response.retained_rank,
                "available_rank": response.available_rank,
                "midpoint_epsilon_p10": float(p10),
                "midpoint_epsilon_p50": float(p50),
                "midpoint_epsilon_p90": float(p90),
                "pair_count": pair_count,
                "pair_seed": pair_seed,
                "latent_relative_shift": "",
                "field_relative_shift": "",
                "field_to_latent_shift_ratio": "",
                "source_artifact": path.as_posix(),
                "source_artifact_sha256": source_hash,
                "checkpoint_sha256": checkpoint_hash,
                "derived_git_commit": derived_git_commit,
            }
            rows.append(row)
            payload_parameterizations[method] = {
                "r2_lin": response.r2,
                "r2_oos_mean": cross_validated.mean,
                "r2_oos_std": cross_validated.std,
                "r2_oos_fold_scores": list(cross_validated.fold_scores),
                "n_folds": n_folds,
                "fold_seed": fold_seed,
                "retained_rank": response.retained_rank,
                "available_rank": response.available_rank,
                "midpoint_epsilon": {
                    "P10": float(p10),
                    "P50": float(p50),
                    "P90": float(p90),
                },
                "source_artifact": path.as_posix(),
                "source_artifact_sha256": source_hash,
                **method_metadata,
            }
            sources[method] = {
                "path": path.as_posix(),
                "sha256": source_hash,
            }

            final_name = _stage_names(handle)[-1]
            final_stage = int(final_name.removeprefix("stage_"))
            if final_stage > 0:
                final_group = handle[final_name]
                final_parameters = np.asarray(
                    final_group["parameters"], dtype=np.float64
                )
                final_simulated = np.asarray(
                    final_group["simulated_data"], dtype=np.float64
                )
                final_response = linear_response_diagnostic(
                    final_parameters,
                    final_simulated,
                    svd_energy=svd_energy,
                )
                final_cross_validated = _cross_validated_linear_response(
                    final_parameters,
                    final_simulated,
                    svd_energy=svd_energy,
                    n_folds=n_folds,
                    seed=fold_seed,
                )
                rows.append(
                    {
                        "strategy": artifact_strategy,
                        "parameterization": method,
                        "diagnostic": "final_stage_linearity_oos",
                        "stage": final_stage,
                        "r2_lin": final_response.r2,
                        "r2_oos_mean": final_cross_validated.mean,
                        "r2_oos_std": final_cross_validated.std,
                        "n_folds": n_folds,
                        "fold_seed": fold_seed,
                        "retained_rank": final_response.retained_rank,
                        "available_rank": final_response.available_rank,
                        "midpoint_epsilon_p10": "",
                        "midpoint_epsilon_p50": "",
                        "midpoint_epsilon_p90": "",
                        "pair_count": "",
                        "pair_seed": "",
                        "latent_relative_shift": "",
                        "field_relative_shift": "",
                        "field_to_latent_shift_ratio": "",
                        "source_artifact": path.as_posix(),
                        "source_artifact_sha256": source_hash,
                        "checkpoint_sha256": checkpoint_hash,
                        "derived_git_commit": derived_git_commit,
                    }
                )
                payload_parameterizations[method]["final_stage_linearity"] = {
                    "stage": final_stage,
                    "r2_lin": final_response.r2,
                    "r2_oos_mean": final_cross_validated.mean,
                    "r2_oos_std": final_cross_validated.std,
                    "r2_oos_fold_scores": list(final_cross_validated.fold_scores),
                    "retained_rank": final_response.retained_rank,
                    "available_rank": final_response.available_rank,
                }

    fm_updates: list[dict[str, object]] = []
    fm_path = inversions["fm"]
    with h5py.File(fm_path) as handle:
        stage_names = _stage_names(handle)
        for previous_name, current_name in zip(
            stage_names, stage_names[1:], strict=False
        ):
            previous = handle[previous_name]
            current = handle[current_name]
            latent_shift = relative_frobenius_shift(
                np.asarray(previous["parameters"], dtype=np.float64),
                np.asarray(current["parameters"], dtype=np.float64),
            )
            field_shift = relative_frobenius_shift(
                np.asarray(previous["logk"], dtype=np.float64),
                np.asarray(current["logk"], dtype=np.float64),
            )
            ratio = field_shift / latent_shift if latent_shift > 0 else float("nan")
            stage = int(current_name.removeprefix("stage_"))
            update: dict[str, object] = {
                "stage": stage,
                "latent_relative_shift": latent_shift,
                "field_relative_shift": field_shift,
                "field_to_latent_shift_ratio": ratio,
            }
            fm_updates.append(update)
            rows.append(
                {
                    "strategy": strategy,
                    "parameterization": "fm",
                    "diagnostic": "update_shift",
                    "stage": stage,
                    "r2_lin": "",
                    "r2_oos_mean": "",
                    "r2_oos_std": "",
                    "n_folds": "",
                    "fold_seed": "",
                    "retained_rank": "",
                    "available_rank": "",
                    "midpoint_epsilon_p10": "",
                    "midpoint_epsilon_p50": "",
                    "midpoint_epsilon_p90": "",
                    "pair_count": "",
                    "pair_seed": "",
                    **update,
                    "source_artifact": fm_path.as_posix(),
                    "source_artifact_sha256": sources["fm"]["sha256"],
                    "checkpoint_sha256": str(
                        metadata.get("fm", {}).get("checkpoint_sha256", "")
                    ),
                    "derived_git_commit": derived_git_commit,
                }
            )
    return rows, {
        "schema_version": 2,
        "strategy": strategy,
        "svd_energy": svd_energy,
        "pair_count": pair_count,
        "pair_seed": pair_seed,
        "n_folds": n_folds,
        "fold_seed": fold_seed,
        "parameterizations": payload_parameterizations,
        "fm_update_shifts": fm_updates,
        "derived_git_commit": derived_git_commit,
    }


def _map_inversions(paths: list[Path]) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    for path in paths:
        with h5py.File(path) as handle:
            method = str(handle.attrs["method"])
        if method in mapped:
            raise ValueError(f"duplicate inversion method: {method}")
        mapped[method] = path
    return mapped


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty diagnostics")
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fm-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--prior-fields", type=Path, required=True)
    parser.add_argument("--inversion", type=Path, action="append", required=True)
    parser.add_argument("--output-table", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--pair-count", type=int, default=200)
    parser.add_argument("--pair-seed", type=int)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--fold-seed", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--git-commit")
    args = parser.parse_args()

    config = load_egg_inversion_config(args.config)
    with h5py.File(args.prior_fields) as handle:
        initial_fields = np.asarray(handle["logk"][: config.ensemble_size], dtype=np.float64)
        active = np.asarray(handle["active_mask"], dtype=bool)
    pca = PCAParameterization.fit(
        initial_fields[:, active], variance_fraction=config.pca_variance_fraction
    )

    def decode_pca(parameters: np.ndarray) -> np.ndarray:
        return embed_active(pca.decode(parameters), active)

    _, decode_fm, fm_metadata = build_flow_adapter(
        initial_fields,
        active,
        checkpoint_path=args.checkpoint,
        training_data=args.training_data,
        fm_config_path=args.fm_config,
        strategy="augmentation",
        device_name=args.device,
        batch_size=args.batch_size,
    )
    decoders: dict[str, Decoder] = {
        "raw": partial(embed_active, active=active),
        "pca": decode_pca,
        "fm": decode_fm,
    }
    rows, payload = _build_diagnostics(
        _map_inversions(args.inversion),
        decoders=decoders,
        svd_energy=config.svd_energy,
        pair_count=args.pair_count,
        pair_seed=config.seed if args.pair_seed is None else args.pair_seed,
        n_folds=args.n_folds,
        fold_seed=config.seed if args.fold_seed is None else args.fold_seed,
        derived_git_commit=args.git_commit or _git_commit(),
        decoder_metadata={"fm": fm_metadata},
    )
    _write_csv(args.output_table, rows)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "row_count": len(rows),
                "output_table": str(args.output_table),
                "output_json": str(args.output_json),
                "device": fm_metadata["device"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
