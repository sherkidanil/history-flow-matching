"""Run one raw, PCA, or FM Egg ES-MDA experiment end to end."""

from __future__ import annotations

import argparse
import json
import subprocess
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Literal

import h5py  # type: ignore[import-untyped]
import numpy as np
from egg_flow_adapter import build_flow_adapter, embed_active

from fmgeo.artifacts import (
    canonical_config_hash,
    create_artifact_record,
    sha256_file,
    update_manifest_atomic,
)
from fmgeo.forward.egg import extract_egg_observations, run_egg_forward
from fmgeo.inverse.egg import (
    EggAssimilationStage,
    load_egg_inversion_config,
    run_egg_esmda,
)
from fmgeo.metrics.uq import coverage, forecast_quantiles
from fmgeo.param.pca import PCAParameterization

Method = Literal["raw", "pca", "fm"]


def _resolve_parameterization_label(method: Method, label: str | None) -> str:
    if label is None:
        return method
    resolved = label.strip()
    if not resolved:
        raise ValueError("parameterization label must be non-empty")
    return resolved


def _write_stage(handle: h5py.File, index: int, stage: EggAssimilationStage) -> None:
    group = handle.create_group(f"stage_{index}")
    group.create_dataset("parameters", data=stage.parameters.astype(np.float32), compression="gzip")
    group.create_dataset("logk", data=stage.fields.astype(np.float32), compression="gzip")
    group.create_dataset(
        "simulated_data", data=stage.forward.simulated_data.astype(np.float32)
    )
    group.create_dataset("fopt", data=stage.forward.fopt.astype(np.float32))
    string_dtype = h5py.string_dtype(encoding="utf-8")
    group.create_dataset("status", data=stage.forward.status.astype(string_dtype))
    group.create_dataset(
        "runtime_seconds", data=stage.forward.runtime_seconds.astype(np.float32)
    )
    group.create_dataset("cache_hit", data=stage.forward.cache_hit)
    group.create_dataset("returncode", data=stage.forward.returncode)
    group.create_dataset("stderr", data=np.asarray(stage.forward.stderr, dtype=string_dtype))
    group.create_dataset(
        "metadata_json",
        data=np.asarray(
            [json.dumps(item, sort_keys=True) for item in stage.forward.metadata],
            dtype=string_dtype,
        ),
    )
    handle.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("raw", "pca", "fm"), required=True)
    parser.add_argument("--parameterization-label")
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fm-config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--training-data", type=Path)
    parser.add_argument("--prior-fields", type=Path, required=True)
    parser.add_argument("--truth-case", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--flow-command", type=Path, required=True)
    parser.add_argument("--simulator-id", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    method: Method = args.method
    parameterization_label = _resolve_parameterization_label(
        method, args.parameterization_label
    )
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    if method == "fm" and not all((args.fm_config, args.checkpoint, args.training_data)):
        raise ValueError("FM method requires fm-config, checkpoint, and training-data")

    config = load_egg_inversion_config(args.config)
    with h5py.File(args.prior_fields) as handle:
        initial_fields = np.asarray(handle["logk"][: config.ensemble_size], dtype=np.float64)
        active = np.asarray(handle["active_mask"], dtype=bool)
    if len(initial_fields) != config.ensemble_size:
        raise ValueError("prior artifact has fewer fields than the configured ensemble")

    parameterization: dict[str, object] = {}
    decode: Callable[[np.ndarray], np.ndarray]
    if method == "raw":
        initial_parameters = initial_fields[:, active]
        decode = partial(embed_active, active=active)
    elif method == "pca":
        pca = PCAParameterization.fit(
            initial_fields[:, active], variance_fraction=config.pca_variance_fraction
        )
        initial_parameters = pca.encode(initial_fields[:, active])

        def decode(parameters: np.ndarray) -> np.ndarray:
            return embed_active(pca.decode(parameters), active)

        parameterization = {
            "pca_rank": pca.rank,
            "pca_explained_variance_fraction": pca.explained_variance_fraction,
        }
    else:
        initial_parameters, decode, parameterization = build_flow_adapter(
            initial_fields,
            active,
            checkpoint_path=args.checkpoint,
            training_data=args.training_data,
            fm_config_path=args.fm_config,
            strategy=args.strategy,
            device_name=args.device,
            batch_size=args.batch_size,
        )

    truth = extract_egg_observations(
        args.truth_case,
        history_end_day=config.history_end_day,
        observation_interval_days=config.observation_interval_days,
        oil_rate_relative_sigma=config.oil_rate_relative_sigma,
        water_rate_relative_sigma=config.water_rate_relative_sigma,
        rate_sigma_floor=config.rate_sigma_floor,
        water_breakthrough_fraction=config.water_breakthrough_fraction,
    )
    truth_data = np.asarray(truth["d"], dtype=np.float64)
    sigma = np.asarray(truth["sigma"], dtype=np.float64)
    observation_seed, assimilation_seed = np.random.SeedSequence(config.seed).spawn(2)
    observation = truth_data + np.random.default_rng(observation_seed).normal(scale=sigma)
    covariance = np.diag(sigma**2)
    evaluator = partial(
        run_egg_forward,
        template_dir=args.template_dir,
        simulator_command=(str(args.flow_command.resolve()),),
        simulator_id=args.simulator_id,
        work_root=args.work_root,
        cache_dir=args.cache_dir,
        history_end_day=config.history_end_day,
        observation_interval_days=config.observation_interval_days,
        oil_rate_relative_sigma=config.oil_rate_relative_sigma,
        water_rate_relative_sigma=config.water_rate_relative_sigma,
        rate_sigma_floor=config.rate_sigma_floor,
        water_breakthrough_fraction=config.water_breakthrough_fraction,
        timeout=config.timeout_seconds,
        min_free_disk_gb=config.min_free_disk_gb,
    )

    provenance = {
        "protocol": config,
        "method": method,
        "strategy": args.strategy,
        "prior_sha256": sha256_file(args.prior_fields),
        "simulator_id": args.simulator_id,
        **parameterization,
    }
    if args.parameterization_label is not None:
        provenance["parameterization_label"] = parameterization_label
    experiment_hash = canonical_config_hash(provenance)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as output:
        output.attrs["config_hash"] = experiment_hash
        output.attrs["method"] = method
        output.attrs["parameterization_label"] = parameterization_label
        output.attrs["strategy"] = args.strategy
        output.create_dataset("truth_data", data=truth_data.astype(np.float32))
        output.create_dataset("observation", data=observation.astype(np.float32))
        output.create_dataset("sigma", data=sigma.astype(np.float32))
        stages = run_egg_esmda(
            initial_parameters,
            decode=decode,
            evaluator=evaluator,
            observation=observation,
            observation_covariance=covariance,
            inflations=config.inflations,
            rng=np.random.default_rng(assimilation_seed),
            workers=config.workers,
            svd_energy=config.svd_energy,
            on_stage=partial(_write_stage, output),
        )

    truth_fopt = float(truth["FOPT_16.5y"])
    stage_reports: list[dict[str, object]] = []
    for index, stage in enumerate(stages):
        quantiles = forecast_quantiles(stage.forward.fopt)
        misfits = np.mean(
            ((stage.forward.simulated_data - observation[None, :]) / sigma[None, :]) ** 2,
            axis=1,
        )
        stage_reports.append(
            {
                "stage": index,
                "fopt": {
                    **quantiles,
                    "truth": truth_fopt,
                    "covered": coverage(
                        truth_fopt, lower=quantiles["P10"], upper=quantiles["P90"]
                    ),
                },
                "mean_normalized_data_misfit": float(np.mean(misfits)),
                "cache_hits": int(stage.forward.cache_hit.sum()),
                "runtime_seconds_sum": float(stage.forward.runtime_seconds.sum()),
            }
        )

    git_commit = args.git_commit or subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    record = create_artifact_record(
        args.output,
        root=Path.cwd(),
        shape=(len(stages), *stages[-1].fields.shape),
        dtype="float32",
        config_hash=experiment_hash,
        git_commit=git_commit,
    )
    update_manifest_atomic(args.manifest, [record])
    report = {
        "benchmark": "Egg",
        "method": method,
        "parameterization_label": parameterization_label,
        "strategy": args.strategy,
        "config_hash": experiment_hash,
        "prior_sha256": sha256_file(args.prior_fields),
        "truth_case": str(args.truth_case),
        "truth_terminal_fopt": truth_fopt,
        "observation_count": len(observation),
        "observation_noise_seed": observation_seed.entropy,
        "parameterization": parameterization,
        "n_sim": sum(len(stage.fields) for stage in stages),
        "n_failed": sum(int(np.sum(stage.forward.status != "ok")) for stage in stages),
        "stages": stage_reports,
        "artifact": record.model_dump(mode="json"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
