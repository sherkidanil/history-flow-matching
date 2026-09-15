"""Train one Egg flow-matching model under the shared controlled budget."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Literal

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
import yaml
from pydantic import Field

from fmgeo.artifacts import (
    canonical_config_hash,
    create_artifact_record,
    sha256_file,
    update_manifest_atomic,
)
from fmgeo.config import StrictModel
from fmgeo.param.flowmatching.model_unet3d import UNet3D
from fmgeo.param.flowmatching.train import (
    LayerTrendNormalizer,
    MaternSourceSampler,
    save_checkpoint_policy,
    train_flow_matching,
)
from fmgeo.runtime import select_device


class EggDataConfig(StrictModel):
    field: Literal["logk"]
    shape_zyx: tuple[int, int, int]
    training_samples: int = Field(ge=1)
    remove_layer_trend: Literal[True]


class SourceConfig(StrictModel):
    kind: Literal["matern"]
    nu: float = Field(gt=0)
    corr_len_cells_zyx: tuple[float, float, float]


class ModelConfig(StrictModel):
    kind: Literal["unet3d"]
    in_channels: Literal[1]
    base_channels: int = Field(ge=1)
    time_dim: int = Field(ge=4)
    coarse_attention_only: Literal[True]


class TrainingConfig(StrictModel):
    batch_size: int = Field(ge=1)
    epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    gradient_clip_norm: float = Field(gt=0)
    ema_decay: float = Field(gt=0, lt=1)
    checkpoint_policy: Literal["ema_and_latest_resume_only"]


class IntegrationConfig(StrictModel):
    method: Literal["euler", "heun"]
    steps: int = Field(ge=1)


class EvaluationConfig(StrictModel):
    reference: Literal["official_heldout"]
    sample_count: int = Field(ge=1)
    metrics: tuple[str, ...]


class EggFMConfig(StrictModel):
    seed: int = Field(ge=0)
    strategies: tuple[Literal["augmentation", "mps", "procedural"], ...]
    data: EggDataConfig
    source: SourceConfig
    model: ModelConfig
    training: TrainingConfig
    integration: IntegrationConfig
    evaluation: EvaluationConfig


def load_fm_config(path: Path) -> EggFMConfig:
    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("FM configuration root must be a mapping")
    return EggFMConfig.model_validate(payload)


def load_training_artifact(
    path: Path, config: EggFMConfig, strategy: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load and validate one strategy artifact against the equal-budget config."""

    with h5py.File(path) as handle:
        stored_strategy = str(handle.attrs.get("strategy", ""))
        if stored_strategy != strategy:
            raise ValueError(f"artifact strategy is {stored_strategy!r}, expected {strategy!r}")
        values = np.asarray(handle[config.data.field], dtype=np.float32)
        active = np.asarray(handle["active_mask"], dtype=bool)
    expected = (config.data.training_samples, *config.data.shape_zyx)
    if values.shape != expected:
        raise ValueError(f"equal-budget artifact must have shape {expected}, found {values.shape}")
    if active.shape != config.data.shape_zyx or not np.any(active):
        raise ValueError("active mask does not match the configured Egg grid")
    if not np.all(np.isfinite(values)):
        raise ValueError("training artifact contains non-finite values")
    targets = torch.from_numpy(values)[:, None]
    mask = torch.from_numpy(active)[None, None]
    return targets, mask


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--strategy", choices=("augmentation", "mps", "procedural"), required=True
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--git-commit")
    args = parser.parse_args()

    config = load_fm_config(args.config)
    if args.strategy not in config.strategies:
        raise ValueError(f"strategy {args.strategy!r} is not declared in the shared config")
    selected_device = select_device(
        args.device,
        cuda_available=torch.cuda.is_available(),
        mps_available=bool(torch.backends.mps.is_available()),
    )
    device = torch.device(selected_device)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    targets, active_mask = load_training_artifact(args.data, config, args.strategy)
    normalizer = LayerTrendNormalizer.fit(targets, active_mask)
    normalized = normalizer.transform(targets)
    normalized = torch.where(active_mask, normalized, torch.zeros_like(normalized))
    model = UNet3D(
        in_channels=config.model.in_channels,
        base_channels=config.model.base_channels,
        time_dim=config.model.time_dim,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    source = MaternSourceSampler(
        shape=config.data.shape_zyx,
        corr_len=config.source.corr_len_cells_zyx,
        nu=config.source.nu,
    )
    losses, ema = train_flow_matching(
        model,
        targets=normalized,
        source_sampler=source,
        active_mask=active_mask,
        optimizer=optimizer,
        epochs=config.training.epochs,
        batch_size=config.training.batch_size,
        seed=config.seed,
        ema_decay=config.training.ema_decay,
        gradient_clip_norm=config.training.gradient_clip_norm,
    )

    config_hash = canonical_config_hash(config)
    data_hash = sha256_file(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    common = {
        "strategy": args.strategy,
        "config_hash": config_hash,
        "data_sha256": data_hash,
        "model_config": config.model.model_dump(mode="json"),
        "normalizer_mean": normalizer.mean,
        "normalizer_scale": normalizer.scale,
    }
    save_checkpoint_policy(
        args.output_dir,
        ema_state={**common, "model": ema.state_dict()},
        resume_state={
            **common,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "completed_epochs": config.training.epochs,
        },
    )

    git_commit = args.git_commit or subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    records = [
        create_artifact_record(
            checkpoint,
            root=Path.cwd(),
            shape=(parameter_count,),
            dtype="torch.float32",
            config_hash=config_hash,
            git_commit=git_commit,
        )
        for checkpoint in (args.output_dir / "ema.pt", args.output_dir / "resume.pt")
    ]
    update_manifest_atomic(args.manifest, records)
    loss_summary = {
        "first": losses[0],
        "last": losses[-1],
        "minimum": min(losses),
    }
    report = {
        "benchmark": "Egg",
        "strategy": args.strategy,
        "config_hash": config_hash,
        "data_sha256": data_hash,
        "device": selected_device,
        "torch_version": torch.__version__,
        "platform": platform.platform(),
        "parameter_count": parameter_count,
        "optimization_steps": len(losses),
        "loss": {**loss_summary, "history": losses},
        "artifacts": [record.model_dump(mode="json") for record in records],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "loss": loss_summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
