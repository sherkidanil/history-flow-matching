"""Validated protocol primitives for the Egg history-matching experiment."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from numpy.typing import ArrayLike, NDArray
from pydantic import Field, model_validator

from fmgeo.config import StrictModel
from fmgeo.forward.egg import EGG_PRODUCERS
from fmgeo.forward.runner import ForwardResult
from fmgeo.inverse.esmda import esmda_update, validate_inflations


class EggLocalizationConfig(StrictModel):
    """Optional cell-to-well covariance localization settings."""

    enabled: bool = False
    kind: Literal["gaspari_cohn"] = "gaspari_cohn"
    radius_m: float = Field(default=64.0, gt=0)
    cell_size_yx_m: tuple[float, float] = (8.0, 8.0)

    @model_validator(mode="after")
    def validate_cell_sizes(self) -> EggLocalizationConfig:
        if any(value <= 0 for value in self.cell_size_yx_m):
            raise ValueError("localization cell sizes must be positive")
        return self


class EggInversionConfig(StrictModel):
    """Fixed scientific and execution settings for the Egg comparison."""

    seed: int = Field(ge=0)
    truth_realization: int = Field(ge=1, le=100)
    ensemble_size: int = Field(ge=2)
    inflations: tuple[float, ...]
    svd_energy: float = Field(gt=0, le=1)
    pca_variance_fraction: float = Field(gt=0, le=1)
    history_end_day: float = Field(gt=0)
    forecast_end_day: float = Field(gt=0)
    observation_interval_days: float = Field(gt=0)
    oil_rate_relative_sigma: float = Field(gt=0)
    water_rate_relative_sigma: float = Field(gt=0)
    rate_sigma_floor: float = Field(gt=0)
    water_breakthrough_fraction: float = Field(gt=0, lt=1)
    high_permeability_quantile: float = Field(gt=0, lt=1)
    workers: int = Field(ge=1)
    timeout_seconds: float = Field(gt=0)
    min_free_disk_gb: float = Field(ge=0)
    localization: EggLocalizationConfig = EggLocalizationConfig()
    fm_latent_pca: bool = False

    @model_validator(mode="after")
    def validate_protocol(self) -> EggInversionConfig:
        validate_inflations(self.inflations)
        if self.forecast_end_day <= self.history_end_day:
            raise ValueError("forecast end must be later than history end")
        intervals = self.history_end_day / self.observation_interval_days
        if not np.isclose(intervals, round(intervals), rtol=0.0, atol=1e-10):
            raise ValueError("history end must be divisible by observation interval")
        return self

    @property
    def observation_count(self) -> int:
        times = int(round(self.history_end_day / self.observation_interval_days))
        return 2 * len(EGG_PRODUCERS) * times


def load_egg_inversion_config(path: str | Path) -> EggInversionConfig:
    """Load a strict Egg inversion configuration."""

    with Path(path).open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("Egg inversion configuration root must be a mapping")
    return EggInversionConfig.model_validate(payload)


@dataclass(frozen=True)
class EggEnsembleForwardResult:
    """Index-stable compact output from an Egg forward ensemble."""

    simulated_data: NDArray[np.float64]
    fopt: NDArray[np.float64]
    status: NDArray[np.str_]
    runtime_seconds: NDArray[np.float64]
    cache_hit: NDArray[np.bool_]
    returncode: NDArray[np.int64]
    stderr: tuple[str, ...]
    metadata: tuple[dict[str, object] | None, ...]


def evaluate_egg_ensemble(
    fields: ArrayLike,
    *,
    evaluator: Callable[[NDArray[np.float64]], ForwardResult],
    workers: int,
) -> EggEnsembleForwardResult:
    """Evaluate fields concurrently without dropping or reordering failures."""

    values = np.asarray(fields, dtype=np.float64)
    if values.ndim != 4 or values.shape[0] == 0 or not np.all(np.isfinite(values)):
        raise ValueError("fields must be a non-empty finite ensemble-first 4D array")
    if workers < 1:
        raise ValueError("workers must be positive")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(evaluator, values))
    observation_sizes = {len(item.d) for item in results if item.status == "ok" and item.d}
    if len(observation_sizes) != 1:
        raise RuntimeError("forward ensemble has no successful common observation dimension")
    observation_count = observation_sizes.pop()
    simulated = np.full((len(results), observation_count), np.nan, dtype=np.float64)
    fopt = np.full(len(results), np.nan, dtype=np.float64)
    for index, result in enumerate(results):
        if result.status == "ok" and result.d is not None and result.fopt_16_5y is not None:
            if len(result.d) != observation_count:
                raise RuntimeError("successful forward members have inconsistent dimensions")
            simulated[index] = result.d
            fopt[index] = result.fopt_16_5y
    return EggEnsembleForwardResult(
        simulated_data=simulated,
        fopt=fopt,
        status=np.asarray([item.status for item in results]),
        runtime_seconds=np.asarray([item.runtime_seconds for item in results]),
        cache_hit=np.asarray([item.cache_hit for item in results]),
        returncode=np.asarray(
            [-999 if item.returncode is None else item.returncode for item in results],
            dtype=np.int64,
        ),
        stderr=tuple(item.stderr for item in results),
        metadata=tuple(item.metadata for item in results),
    )


@dataclass(frozen=True)
class EggAssimilationStage:
    """One complete, index-stable ES-MDA stage."""

    parameters: NDArray[np.float64]
    fields: NDArray[np.float64]
    forward: EggEnsembleForwardResult


def run_egg_esmda(
    initial_parameters: ArrayLike,
    *,
    decode: Callable[[NDArray[np.float64]], NDArray[np.float64]],
    evaluator: Callable[[NDArray[np.float64]], ForwardResult],
    observation: ArrayLike,
    observation_covariance: ArrayLike,
    inflations: Sequence[float],
    rng: np.random.Generator,
    workers: int,
    svd_energy: float,
    localization: ArrayLike | None = None,
    on_stage: Callable[[int, EggAssimilationStage], None] | None = None,
) -> tuple[EggAssimilationStage, ...]:
    """Run all ES-MDA stages, refusing to silently discard failed members."""

    inflation_schedule = validate_inflations(inflations)
    parameters = np.asarray(initial_parameters, dtype=np.float64)
    if parameters.ndim != 2 or parameters.shape[0] < 2 or not np.all(np.isfinite(parameters)):
        raise ValueError("initial parameters must be a finite ensemble-first matrix")
    stages: list[EggAssimilationStage] = []
    for stage_index in range(len(inflation_schedule) + 1):
        fields = np.asarray(decode(parameters), dtype=np.float64)
        if fields.ndim != 4 or fields.shape[0] != len(parameters):
            raise ValueError("decoded fields must have ensemble-first 4D shape")
        forward = evaluate_egg_ensemble(fields, evaluator=evaluator, workers=workers)
        failed = np.flatnonzero(forward.status != "ok")
        stage = EggAssimilationStage(
            parameters=parameters.copy(),
            fields=fields.copy(),
            forward=forward,
        )
        stages.append(stage)
        if on_stage is not None:
            on_stage(stage_index, stage)
        if len(failed):
            raise RuntimeError(
                f"ES-MDA stage {stage_index} retained {len(failed)} failed members; "
                "rerun or diagnose them before updating"
            )
        if stage_index == len(inflation_schedule):
            break
        parameters = esmda_update(
            parameters,
            forward.simulated_data,
            observation=observation,
            observation_covariance=observation_covariance,
            inflation=inflation_schedule[stage_index],
            rng=rng,
            localization=localization,
            svd_energy=svd_energy,
        )
    return tuple(stages)
