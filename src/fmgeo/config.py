"""Strict, path-stable experiment configuration."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class StrictModel(BaseModel):
    """Base model that rejects misspelled or obsolete settings."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ExecutionConfig(StrictModel):
    """Capabilities and limits for one execution environment."""

    profile: Literal["local", "cluster"]
    workers: int = Field(ge=1)
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    min_free_disk_gb: float = Field(ge=0)
    work_dir: Path
    flow_command: str = Field(min_length=1)


class EsmdaConfig(StrictModel):
    """Core ensemble smoother settings."""

    ensemble_size: int = Field(ge=2)
    inflations: tuple[float, ...]

    @field_validator("inflations")
    @classmethod
    def validate_inflations(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if not values or any(value <= 0 or not math.isfinite(value) for value in values):
            raise ValueError("inflations must be finite positive values")
        if not math.isclose(sum(1.0 / value for value in values), 1.0, abs_tol=1e-8):
            raise ValueError("reciprocal inflation factors must sum to one")
        return values


class ProjectConfig(StrictModel):
    """Validated top-level experiment configuration."""

    seed: int = Field(ge=0)
    scale: Literal["smoke", "publication"]
    execution: ExecutionConfig
    esmda: EsmdaConfig


def load_config(path: str | Path) -> ProjectConfig:
    """Load YAML and resolve its paths relative to the configuration file."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be a mapping")

    execution = payload.get("execution")
    if isinstance(execution, dict) and isinstance(execution.get("work_dir"), str):
        work_dir = Path(execution["work_dir"]).expanduser()
        if not work_dir.is_absolute():
            work_dir = config_path.parent / work_dir
        execution["work_dir"] = work_dir.resolve()

    return ProjectConfig.model_validate(payload)

