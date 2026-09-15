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
from m8_train_egg import load_fm_config
from pydantic import Field

from fmgeo.artifacts import (
    canonical_config_hash,
    create_artifact_record,
    sha256_file,
    update_manifest_atomic,
)
from fmgeo.config import StrictModel
from fmgeo.metrics.egg import egg_distribution_metrics
from fmgeo.param.flowmatching.model_unet3d import UNet3D
from fmgeo.param.flowmatching.sample import FlowTransform
from fmgeo.param.flowmatching.train import LayerTrendNormalizer, MaternSourceSampler
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--strategy-config", type=Path, required=True)
    parser.add_argument("--metric-config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--realizations-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")

    config = load_fm_config(args.config)
    metric_config = load_metric_config(args.metric_config)
    strategy_config = load_strategy_config(args.strategy_config)
    strategy = strategy_config.strategy
    training_config_hash = canonical_config_hash(config)
    metric_config_hash = canonical_config_hash(metric_config)
    sample_config_hash = canonical_config_hash({"training": config, "metrics": metric_config})
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
    model = UNet3D(
        in_channels=config.model.in_channels,
        base_channels=config.model.base_channels,
        time_dim=config.model.time_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    normalizer = LayerTrendNormalizer(
        mean=checkpoint["normalizer_mean"], scale=checkpoint["normalizer_scale"]
    )

    ensemble = load_egg_ensemble(args.realizations_dir)
    heldout = np.asarray([index - 1 for index in strategy_config.heldout_indices])
    reference = ensemble[heldout]
    training = np.delete(ensemble, heldout, axis=0)
    with h5py.File(args.training_data) as handle:
        active = np.asarray(handle["active_mask"], dtype=bool)
    if active.shape != EGG_SHAPE:
        raise ValueError("checkpoint data active mask does not match Egg")
    threshold = float(np.quantile(training[:, active], metric_config.high_permeability_quantile))

    mask = torch.from_numpy(active)[None, None].to(device)
    source_sampler = MaternSourceSampler(
        shape=config.data.shape_zyx,
        corr_len=config.source.corr_len_cells_zyx,
        nu=config.source.nu,
    )
    transform = FlowTransform(
        model,
        steps=config.integration.steps,
        method=config.integration.method,
        mask=mask,
    )
    generator = torch.Generator(device=device).manual_seed(config.seed + 10_000)
    generated_batches: list[np.ndarray] = []
    roundtrip_relative_error: float | None = None
    remaining = config.evaluation.sample_count
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
        completed = config.evaluation.sample_count - remaining
        print(f"generated {completed}/{config.evaluation.sample_count}")
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
