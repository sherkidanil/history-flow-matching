"""Prepare auditable, equal-size Egg training datasets for three strategies."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Literal

import h5py  # type: ignore[import-untyped]
import numpy as np
import yaml
from pydantic import Field, model_validator

from fmgeo.artifacts import (
    canonical_config_hash,
    create_artifact_record,
    sha256_file,
    write_manifest_atomic,
)
from fmgeo.config import StrictModel
from fmgeo.grids import parse_numeric_keyword
from fmgeo.priors.egg_augment import augment_crops, directional_rotation_allowed
from fmgeo.priors.egg_mps import generate_mps_realizations
from fmgeo.priors.egg_procedural import generate_procedural_egg

EGG_SHAPE = (7, 60, 60)
EGG_REALIZATION_COUNT = 100


class EggStrategyConfig(StrictModel):
    """Strict common schema for an Egg training-data strategy."""

    strategy: Literal["augmentation", "mps", "procedural"]
    seed: int = Field(ge=0)
    dataset_count: int = Field(ge=1)
    heldout_indices: tuple[int, ...]
    crop_shape_zyx: tuple[int, int, int] | None = None
    flip_axes_zyx: tuple[bool, bool, bool] | None = None
    rotations: bool | None = None
    rotation_variogram_tolerance: float | None = Field(default=None, ge=0)
    channel_quantile: float | None = Field(default=None, gt=0, lt=1)
    channel_width_cells: int | None = Field(default=None, ge=1)
    channel_count: int | None = Field(default=None, ge=1)
    backend: Literal["mpslib"] | None = None
    method: Literal["mps_genesim", "mps_snesim_tree"] | None = None
    conditioning_nodes: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> EggStrategyConfig:
        required: dict[str, tuple[str, ...]] = {
            "augmentation": (
                "crop_shape_zyx",
                "flip_axes_zyx",
                "rotations",
                "rotation_variogram_tolerance",
            ),
            "mps": ("channel_quantile", "backend", "method", "conditioning_nodes"),
            "procedural": ("channel_quantile", "channel_width_cells", "channel_count"),
        }
        missing = [name for name in required[self.strategy] if getattr(self, name) is None]
        if missing:
            raise ValueError(f"{self.strategy} config is missing fields: {missing}")
        if not self.heldout_indices or len(set(self.heldout_indices)) != len(self.heldout_indices):
            raise ValueError("heldout_indices must be non-empty and unique")
        if any(index < 1 or index > EGG_REALIZATION_COUNT for index in self.heldout_indices):
            raise ValueError("heldout_indices must be one-based values between 1 and 100")
        return self


def load_strategy_config(path: Path) -> EggStrategyConfig:
    """Load a strict strategy configuration."""

    with path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("strategy configuration root must be a mapping")
    return EggStrategyConfig.model_validate(payload)


def load_egg_ensemble(realizations_dir: Path) -> np.ndarray:
    """Load all official PERMX fields in numeric realization order as log-k."""

    fields: list[np.ndarray] = []
    for index in range(1, EGG_REALIZATION_COUNT + 1):
        path = realizations_dir / f"PERM{index}_ECL.INC"
        values = np.asarray(parse_numeric_keyword(path, "PERMX"), dtype=np.float32)
        if values.size != int(np.prod(EGG_SHAPE)):
            raise ValueError(f"{path} has {values.size} values, expected {np.prod(EGG_SHAPE)}")
        if np.any(values <= 0) or not np.all(np.isfinite(values)):
            raise ValueError(f"{path} contains invalid permeability")
        fields.append(np.log(values).reshape(EGG_SHAPE).astype(np.float32))
    return np.stack(fields)


def _training_split(ensemble: np.ndarray, heldout_indices: tuple[int, ...]) -> np.ndarray:
    heldout_zero_based = {index - 1 for index in heldout_indices}
    return np.stack(
        [field for index, field in enumerate(ensemble) if index not in heldout_zero_based]
    )


def _calibration(
    training: np.ndarray, active: np.ndarray, quantile: float
) -> tuple[float, float, float, float, float]:
    active_values = training[:, active]
    threshold = float(np.quantile(active_values, quantile))
    background = active_values[active_values < threshold]
    channel = active_values[active_values >= threshold]
    return (
        threshold,
        float(background.mean()),
        float(background.std()),
        float(channel.mean()),
        float(channel.std()),
    )


def _prepare(
    config: EggStrategyConfig,
    ensemble: np.ndarray,
    active: np.ndarray,
    *,
    mpslib_executable_dir: Path | None = None,
) -> tuple[np.ndarray, dict[str, object]]:
    training = _training_split(ensemble, config.heldout_indices)
    if config.strategy == "augmentation":
        assert config.crop_shape_zyx is not None
        assert config.rotations is not None
        assert config.rotation_variogram_tolerance is not None
        assert config.flip_axes_zyx is not None
        generated, metadata = augment_crops(
            training,
            crop_shape=config.crop_shape_zyx,
            count=config.dataset_count,
            seed=config.seed,
            rotations=config.rotations,
            rotation_variogram_tolerance=config.rotation_variogram_tolerance,
            flip_axes_zyx=config.flip_axes_zyx,
        )
        if generated.shape[1:] != EGG_SHAPE:
            raise ValueError("publication Egg FM datasets must retain the full Egg grid shape")
        generated[:, ~active] = 0.0
        return np.asarray(generated, dtype=np.float32), {
            "rotations_accepted": directional_rotation_allowed(
                training,
                relative_tolerance=config.rotation_variogram_tolerance,
            ),
            "transform_counts": {
                str(quarter): sum(item["rotation_quarters"] == quarter for item in metadata)
                for quarter in range(4)
            },
        }

    assert config.channel_quantile is not None
    threshold, background_mean, background_std, channel_mean, channel_std = _calibration(
        training, active, config.channel_quantile
    )
    calibration = {
        "facies_threshold_logk": threshold,
        "background_logk_mean": background_mean,
        "background_logk_std": background_std,
        "channel_logk_mean": channel_mean,
        "channel_logk_std": channel_std,
    }
    if config.strategy == "mps":
        assert config.method is not None
        assert config.conditioning_nodes is not None
        facies_training = np.asarray(training >= threshold, dtype=np.uint8)
        facies = generate_mps_realizations(
            facies_training,
            count=config.dataset_count,
            seed=config.seed,
            method=config.method,
            conditioning_nodes=config.conditioning_nodes,
            executable_dir=mpslib_executable_dir,
        )
        rng = np.random.default_rng(config.seed)
        generated = rng.normal(background_mean, background_std, facies.shape).astype(np.float32)
        channel_cells = facies == 1
        generated[channel_cells] = rng.normal(
            channel_mean, channel_std, int(np.count_nonzero(channel_cells))
        )
        generated[:, ~active] = 0.0
        return generated, {"calibration": calibration, "backend": config.backend}

    assert config.channel_width_cells is not None
    assert config.channel_count is not None
    generated = np.empty((config.dataset_count, *EGG_SHAPE), dtype=np.float32)
    spanning_count = 0
    for index in range(config.dataset_count):
        result = generate_procedural_egg(
            active,
            seed=config.seed + index,
            channel_width_cells=config.channel_width_cells,
            channel_count=config.channel_count,
            background_logk_mean=background_mean,
            background_logk_std=background_std,
            channel_logk_mean=channel_mean,
            channel_logk_std=channel_std,
        )
        generated[index] = result.logk
        spanning_count += int(result.has_spanning_channel)
    return generated, {
        "calibration": calibration,
        "spanning_fraction": spanning_count / config.dataset_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--realizations-dir", type=Path, required=True)
    parser.add_argument("--actnum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--mpslib-executable-dir",
        type=Path,
        help="directory containing a validated source-built MPSlib executable",
    )
    args = parser.parse_args()

    config = load_strategy_config(args.config)
    ensemble = load_egg_ensemble(args.realizations_dir)
    active_values = np.asarray(parse_numeric_keyword(args.actnum, "ACTNUM"), dtype=bool)
    if active_values.size != int(np.prod(EGG_SHAPE)):
        raise ValueError("Egg ACTNUM does not match the 7x60x60 grid")
    active = active_values.reshape(EGG_SHAPE)
    if int(active.sum()) != 18553:
        raise ValueError("official Egg grid must contain 18553 active cells")

    generated, diagnostics = _prepare(
        config,
        ensemble,
        active,
        mpslib_executable_dir=args.mpslib_executable_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output, "w") as handle:
        handle.create_dataset("logk", data=generated, compression="gzip", shuffle=True)
        handle.create_dataset("active_mask", data=active, compression="gzip")
        handle.attrs["strategy"] = config.strategy
        handle.attrs["seed"] = config.seed

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    config_hash = canonical_config_hash(config)
    record = create_artifact_record(
        args.output,
        root=Path.cwd(),
        shape=generated.shape,
        dtype=str(generated.dtype),
        config_hash=config_hash,
        git_commit=git_commit,
    )
    write_manifest_atomic(args.manifest, [record])
    report = {
        "benchmark": "Egg",
        "strategy": config.strategy,
        "training_realizations": EGG_REALIZATION_COUNT - len(config.heldout_indices),
        "heldout_indices": config.heldout_indices,
        "dataset_count": config.dataset_count,
        "config_hash": config_hash,
        "input_hashes": {
            str(args.actnum): sha256_file(args.actnum),
            **{
                str(path): sha256_file(path)
                for path in sorted(args.realizations_dir.glob("PERM*_ECL.INC"))
            },
        },
        "artifact": record.model_dump(mode="json"),
        "diagnostics": diagnostics,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
