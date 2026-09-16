"""Sample and evaluate one Egg FM model with the shared metric schema."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
import yaml
from m8_egg_case import EGG_SHAPE, load_egg_ensemble, load_strategy_config
from m8_train_egg import EggFMConfig, load_fm_config
from pydantic import Field

from fmgeo.artifacts import (
    canonical_config_hash,
    create_artifact_record,
    sha256_file,
    update_manifest_atomic,
)
from fmgeo.config import StrictModel
from fmgeo.metrics.egg import egg_distribution_metrics
from fmgeo.param.flowmatching.models import build_velocity_model
from fmgeo.param.flowmatching.sample import FlowTransform
from fmgeo.param.flowmatching.train import LayerTrendNormalizer, make_source_sampler
from fmgeo.resolution import average_pool_horizontal, scale_correlation_lengths
from fmgeo.runtime import select_device


class EggMetricConfig(StrictModel):
    high_permeability_quantile: float = Field(gt=0, lt=1)
    max_sample_cells: int = Field(ge=2)
    max_sample_fields: int = Field(ge=1)
    max_variogram_lag: int = Field(ge=1)


def load_metric_config(path: Path) -> EggMetricConfig:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("metric configuration root must be a mapping")
    return EggMetricConfig.model_validate(payload)


def _resolve_evaluation_budget(
    config: EggFMConfig,
    *,
    integration_steps: int | None,
    sample_count: int | None,
) -> tuple[int, int]:
    steps = config.integration.steps if integration_steps is None else integration_steps
    count = config.evaluation.sample_count if sample_count is None else sample_count
    if steps <= 0 or count <= 0:
        raise ValueError("evaluation integration steps and sample count must be positive")
    return steps, count


def _checkpoint_epoch(
    path: Path, checkpoint: dict[str, object], configured_epochs: int
) -> int:
    stored = checkpoint.get("completed_epochs")
    if isinstance(stored, int) and not isinstance(stored, bool):
        return stored
    if stored is not None:
        raise ValueError("checkpoint completed_epochs must be an integer")
    prefix = "ema-epoch-"
    if path.stem.startswith(prefix):
        return int(path.stem.removeprefix(prefix))
    return configured_epochs


def _match_reference_resolution(
    fields: np.ndarray,
    *,
    full_active: np.ndarray,
    target_active: np.ndarray,
) -> np.ndarray:
    if fields.shape[1:] == target_active.shape:
        if full_active.shape != target_active.shape or not np.array_equal(
            full_active, target_active
        ):
            raise ValueError("full-resolution active masks do not match")
        return fields
    if fields.shape[1] != target_active.shape[0]:
        raise ValueError("target resolution must preserve the Egg layers")
    ratios = (
        fields.shape[-2] / target_active.shape[-2],
        fields.shape[-1] / target_active.shape[-1],
    )
    if ratios[0] != ratios[1] or not ratios[0].is_integer():
        raise ValueError("target resolution must be an integer isotropic horizontal pooling")
    pooled, pooled_active = average_pool_horizontal(
        fields, active_mask=full_active, factor=int(ratios[0])
    )
    if not np.array_equal(pooled_active, target_active):
        raise ValueError("pooled official active mask does not match the target artifact")
    return pooled


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--strategy-config", type=Path, required=True)
    parser.add_argument("--metric-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--target-active-source", type=Path)
    parser.add_argument("--realizations-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--integration-steps", type=int)
    parser.add_argument("--sample-count", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    config = load_fm_config(args.config)
    integration_steps, sample_count = _resolve_evaluation_budget(
        config,
        integration_steps=args.integration_steps,
        sample_count=args.sample_count,
    )
    metric_config = load_metric_config(args.metric_config)
    strategy_config = load_strategy_config(args.strategy_config)
    strategy = strategy_config.strategy
    training_config_hash = canonical_config_hash(config)
    metric_config_hash = canonical_config_hash(metric_config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint["strategy"] != strategy or checkpoint["config_hash"] != training_config_hash:
        raise ValueError("checkpoint strategy or configuration hash does not match")
    if checkpoint["data_sha256"] != sha256_file(args.training_data):
        raise ValueError("checkpoint training-data hash does not match")

    selected_device = select_device(
        args.device,
        cuda_available=torch.cuda.is_available(),
        mps_available=bool(torch.backends.mps.is_available()),
    )
    device = torch.device(selected_device)
    model = build_velocity_model(config.model.model_dump(mode="python")).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalizer = LayerTrendNormalizer(
        mean=checkpoint["normalizer_mean"], scale=checkpoint["normalizer_scale"]
    )

    ensemble = load_egg_ensemble(args.realizations_dir)
    heldout = np.asarray([index - 1 for index in strategy_config.heldout_indices])
    target_active_source = args.target_active_source or args.training_data
    with h5py.File(target_active_source) as handle:
        active = np.asarray(handle["active_mask"], dtype=bool)
    with h5py.File(args.training_data) as handle:
        training_active = np.asarray(handle["active_mask"], dtype=bool)
    if training_active.shape != config.data.shape_zyx:
        raise ValueError("training-data active mask does not match the configuration")
    if active.shape not in (config.data.shape_zyx, EGG_SHAPE):
        raise ValueError("evaluation must use the training or official full Egg resolution")
    full_active = np.any(ensemble != 0.0, axis=0)
    reference = _match_reference_resolution(
        ensemble[heldout], full_active=full_active, target_active=active
    )
    training = _match_reference_resolution(
        np.delete(ensemble, heldout, axis=0),
        full_active=full_active,
        target_active=active,
    )
    source_corr_len = scale_correlation_lengths(
        config.source.corr_len_cells_zyx,
        reference_shape=config.data.shape_zyx,
        target_shape=active.shape,
    )
    sample_config_hash = canonical_config_hash(
        {
            "training": config,
            "metrics": metric_config,
            "integration_steps": integration_steps,
            "sample_count": sample_count,
            "evaluation_shape": active.shape,
            "source_corr_len_cells_zyx": source_corr_len,
        }
    )
    threshold = float(np.quantile(training[:, active], metric_config.high_permeability_quantile))

    mask = torch.from_numpy(active)[None, None].to(device)
    source_sampler = make_source_sampler(
        config.source.kind,
        shape=active.shape,
        corr_len=source_corr_len,
        nu=config.source.nu,
        corr_len_scale=config.source.corr_len_scale,
    )
    transform = FlowTransform(
        model,
        steps=integration_steps,
        method=config.integration.method,
        mask=mask,
    )
    generator = torch.Generator(device=device).manual_seed(config.seed + 10_000)
    generated_batches: list[np.ndarray] = []
    roundtrip_relative_error: float | None = None
    remaining = sample_count
    while remaining:
        batch_count = min(args.batch_size, remaining)
        source = source_sampler(batch_count, generator, device, torch.float32)
        normalized = transform.forward(source)
        if roundtrip_relative_error is None:
            reconstructed = transform.inverse(normalized[: min(4, batch_count)])
            initial = source[: len(reconstructed)]
            active_batch = torch.broadcast_to(mask, initial.shape)
            numerator = torch.linalg.vector_norm((reconstructed - initial)[active_batch])
            denominator = torch.linalg.vector_norm(initial[active_batch]).clamp_min(1e-12)
            roundtrip_relative_error = float((numerator / denominator).cpu())
        physical = normalizer.inverse(normalized).cpu()
        physical = torch.where(mask.cpu(), physical, torch.zeros_like(physical))
        generated_batches.append(physical[:, 0].numpy())
        remaining -= batch_count
        completed = sample_count - remaining
        print(f"generated {completed}/{sample_count}")
    generated = np.concatenate(generated_batches).astype(np.float32, copy=False)

    metrics = egg_distribution_metrics(
        generated,
        reference,
        active_mask=active,
        high_permeability_threshold=threshold,
        seed=config.seed,
        max_sample_cells=metric_config.max_sample_cells,
        max_sample_fields=metric_config.max_sample_fields,
        max_variogram_lag=metric_config.max_variogram_lag,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.create_dataset("logk", data=generated, compression="gzip", shuffle=True)
        handle.create_dataset("active_mask", data=active, compression="gzip")
        handle.attrs["strategy"] = strategy
        handle.attrs["config_hash"] = sample_config_hash
        handle.attrs["training_shape_zyx"] = config.data.shape_zyx
        handle.attrs["evaluation_shape_zyx"] = active.shape

    git_commit = (
        args.git_commit
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    record = create_artifact_record(
        args.output,
        root=Path.cwd(),
        shape=generated.shape,
        dtype=str(generated.dtype),
        config_hash=sample_config_hash,
        git_commit=git_commit,
    )
    update_manifest_atomic(args.manifest, [record])
    report = {
        "benchmark": "Egg",
        "strategy": strategy,
        "checkpoint": str(args.checkpoint),
        "training_config_hash": training_config_hash,
        "metric_config_hash": metric_config_hash,
        "sample_config_hash": sample_config_hash,
        "checkpoint_epoch": _checkpoint_epoch(
            args.checkpoint, checkpoint, config.training.epochs
        ),
        "integration_steps": integration_steps,
        "sample_count": sample_count,
        "training_shape_zyx": list(config.data.shape_zyx),
        "evaluation_shape_zyx": list(active.shape),
        "source_corr_len_cells_zyx": list(source_corr_len),
        "high_permeability_threshold_logk": threshold,
        "roundtrip_relative_error": roundtrip_relative_error,
        "metrics": metrics,
        "artifact": record.model_dump(mode="json"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
