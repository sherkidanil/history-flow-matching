"""Train one Egg flow-matching model under the shared controlled budget."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Annotated, Literal

import h5py  # type: ignore[import-untyped]
import numpy as np
import torch
import yaml
from pydantic import Field, model_validator

from fmgeo.artifacts import (
    canonical_config_hash,
    create_artifact_record,
    sha256_file,
    update_manifest_atomic,
)
from fmgeo.config import StrictModel
from fmgeo.param.flowmatching.models import build_velocity_model
from fmgeo.param.flowmatching.train import (
    ExponentialMovingAverage,
    LayerTrendNormalizer,
    make_source_sampler,
    save_checkpoint_policy,
    save_ema_snapshot,
    train_flow_matching,
)
from fmgeo.runtime import select_device


class EggDataConfig(StrictModel):
    field: Literal["logk"]
    shape_zyx: tuple[int, int, int]
    training_samples: int = Field(ge=1)
    remove_layer_trend: Literal[True]


class SourceConfig(StrictModel):
    kind: Literal["white", "matern"]
    nu: float = Field(gt=0)
    corr_len_cells_zyx: tuple[float, float, float]
    corr_len_scale: float = Field(default=1.0, gt=0)


class UNetModelConfig(StrictModel):
    kind: Literal["unet3d"]
    in_channels: Literal[1]
    base_channels: int = Field(ge=1)
    time_dim: int = Field(ge=4)
    coarse_attention_only: Literal[True]


class UNOModelConfig(StrictModel):
    kind: Literal["uno3d"]
    in_channels: Literal[1]
    hidden_channels: int = Field(ge=1)
    time_dim: int = Field(ge=4)
    modes_zyx: tuple[int, int, int]
    blocks: int = Field(ge=1)


ModelConfig = Annotated[UNetModelConfig | UNOModelConfig, Field(discriminator="kind")]


class TrainingConfig(StrictModel):
    batch_size: int = Field(ge=1)
    epochs: int = Field(ge=1)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    gradient_clip_norm: float = Field(gt=0)
    ema_decay: float = Field(gt=0, lt=1)
    checkpoint_policy: Literal[
        "ema_and_latest_resume_only", "ema_snapshots_and_latest_resume"
    ]
    evaluation_epochs: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_snapshots(self) -> TrainingConfig:
        requested = self.evaluation_epochs
        if tuple(sorted(set(requested))) != requested:
            raise ValueError("evaluation epochs must be unique and increasing")
        if any(epoch < 1 or epoch > self.epochs for epoch in requested):
            raise ValueError("evaluation epochs must lie inside the training run")
        snapshots = self.checkpoint_policy == "ema_snapshots_and_latest_resume"
        if snapshots != bool(requested):
            raise ValueError("snapshot policy and evaluation epochs must be enabled together")
        return self


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
    model = build_velocity_model(config.model.model_dump(mode="python")).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    source = make_source_sampler(
        config.source.kind,
        shape=config.data.shape_zyx,
        corr_len=config.source.corr_len_cells_zyx,
        nu=config.source.nu,
        corr_len_scale=config.source.corr_len_scale,
    )
    config_hash = canonical_config_hash(config)
    data_hash = sha256_file(args.data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for obsolete in args.output_dir.glob("ema-epoch-*.pt"):
        obsolete.unlink()
    common = {
        "strategy": args.strategy,
        "config_hash": config_hash,
        "data_sha256": data_hash,
        "model_config": config.model.model_dump(mode="json"),
        "normalizer_mean": normalizer.mean,
        "normalizer_scale": normalizer.scale,
    }

    def snapshot_epoch(epoch: int, ema: ExponentialMovingAverage) -> None:
        if epoch in config.training.evaluation_epochs:
            save_ema_snapshot(
                args.output_dir,
                epoch=epoch,
                state={**common, "model": ema.state_dict(), "completed_epochs": epoch},
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
        on_epoch=snapshot_epoch,
    )

    save_checkpoint_policy(
        args.output_dir,
        ema_state={
            **common,
            "model": ema.state_dict(),
            "completed_epochs": config.training.epochs,
        },
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
        for checkpoint in sorted(args.output_dir.glob("*.pt"))
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
        "evaluation_epochs": list(config.training.evaluation_epochs),
        "loss": {**loss_summary, "history": losses},
        "artifacts": [record.model_dump(mode="json") for record in records],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({**report, "loss": loss_summary}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
