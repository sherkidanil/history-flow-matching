"""Egg-specific summary extraction for history matching."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from fmgeo.forward.observables import read_summary_vectors
from fmgeo.forward.runner import ForwardResult, run_simulator
from fmgeo.priors.egg_io import EGG_SHAPE_ZYX, write_egg_permeability_include

EGG_PRODUCERS = ("PROD1", "PROD2", "PROD3", "PROD4")


def parse_egg_well_locations(deck: str) -> dict[str, tuple[int, int]]:
    """Read Egg WELSPECS I/J locations as zero-based ``(y, x)`` pairs."""

    section = re.search(r"(?ims)^\s*WELSPECS\s*$\s*(.*?)^\s*/\s*$", deck)
    if section is None:
        raise ValueError("deck does not contain a terminated WELSPECS section")
    records = re.findall(
        r"(?im)'(INJECT\d+|PROD\d+)'\s+'[^']+'\s+(\d+)\s+(\d+)", section.group(1)
    )
    if not records:
        raise ValueError("WELSPECS contains no Egg injector or producer records")
    locations = {name: (int(j) - 1, int(i) - 1) for name, i, j in records}
    if len(locations) != len(records):
        raise ValueError("WELSPECS contains duplicate Egg well names")
    return locations


def _history_indices(
    times: np.ndarray, *, history_end_day: float, observation_interval_days: float
) -> np.ndarray:
    if history_end_day <= 0 or observation_interval_days <= 0:
        raise ValueError("history end and observation interval must be positive")
    count = history_end_day / observation_interval_days
    if not np.isclose(count, round(count), rtol=0.0, atol=1e-10):
        raise ValueError("history end must be divisible by the observation interval")
    requested = np.arange(1, int(round(count)) + 1, dtype=np.float64)
    requested *= observation_interval_days
    indices: list[int] = []
    for target in requested:
        matches = np.flatnonzero(np.isclose(times, target, rtol=0.0, atol=1e-8))
        if len(matches) != 1:
            raise ValueError(f"summary schedule has no unique output at day {target:g}")
        indices.append(int(matches[0]))
    return np.asarray(indices, dtype=np.int64)


def extract_egg_observations(
    case_path: str | Path,
    *,
    producers: Sequence[str] = EGG_PRODUCERS,
    history_end_day: float,
    observation_interval_days: float,
    oil_rate_relative_sigma: float,
    water_rate_relative_sigma: float,
    rate_sigma_floor: float,
    water_breakthrough_fraction: float = 0.05,
) -> dict[str, Any]:
    """Extract rate observations and their declared independent errors."""

    if not producers or any(not producer for producer in producers):
        raise ValueError("at least one producer is required")
    if oil_rate_relative_sigma <= 0 or water_rate_relative_sigma <= 0:
        raise ValueError("relative rate errors must be positive")
    if rate_sigma_floor <= 0:
        raise ValueError("rate error floor must be positive")
    if not 0 < water_breakthrough_fraction < 1:
        raise ValueError("water breakthrough fraction must be in (0, 1)")
    oil_keys = [f"WOPR:{producer}" for producer in producers]
    water_keys = [f"WWPR:{producer}" for producer in producers]
    keys = ["TIME", "FOPT", *oil_keys, *water_keys]
    vectors = read_summary_vectors(case_path, keys)
    times = vectors["TIME"]
    indices = _history_indices(
        times,
        history_end_day=history_end_day,
        observation_interval_days=observation_interval_days,
    )
    oil = np.concatenate([vectors[key][indices] for key in oil_keys])
    water = np.concatenate([vectors[key][indices] for key in water_keys])
    observations = np.concatenate([oil, water])
    sigma = np.concatenate(
        [
            np.maximum(np.abs(oil) * oil_rate_relative_sigma, rate_sigma_floor),
            np.maximum(np.abs(water) * water_rate_relative_sigma, rate_sigma_floor),
        ]
    )
    water_cut: dict[str, list[float]] = {}
    breakthrough_day: dict[str, float | None] = {}
    for producer, oil_key, water_key in zip(producers, oil_keys, water_keys, strict=True):
        oil_rate = vectors[oil_key]
        water_rate = vectors[water_key]
        total_rate = oil_rate + water_rate
        cut = np.divide(
            water_rate,
            total_rate,
            out=np.zeros_like(water_rate),
            where=np.abs(total_rate) > np.finfo(np.float64).eps,
        )
        hits = np.flatnonzero(cut >= water_breakthrough_fraction)
        water_cut[producer] = cut.tolist()
        breakthrough_day[producer] = None if len(hits) == 0 else float(times[hits[0]])
    return {
        "d": observations,
        "sigma": sigma,
        "times": times[indices],
        # ForwardResult retains the PUNQ-oriented historical field name. For
        # Egg this value is the terminal (10-year) FOPT from the supplied deck.
        "FOPT_16.5y": float(vectors["FOPT"][-1]),
        "metadata": {
            "forecast_times_days": times.tolist(),
            "water_cut": water_cut,
            "water_breakthrough_day": breakthrough_day,
        },
    }


def _egg_cache_key(
    logk: np.ndarray,
    *,
    template_dir: Path,
    simulator_command: Sequence[str],
    simulator_id: str,
    observation_settings: dict[str, object],
) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(logk, dtype=np.float32).tobytes())
    for path in sorted(item for item in template_dir.rglob("*") if item.is_file()):
        digest.update(path.relative_to(template_dir).as_posix().encode())
        digest.update(path.read_bytes())
    metadata = {
        "simulator_command": list(simulator_command),
        "simulator_id": simulator_id,
        "observation_settings": observation_settings,
    }
    digest.update(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode())
    return digest.hexdigest()


def run_egg_forward(
    logk: np.ndarray,
    *,
    template_dir: str | Path,
    simulator_command: Sequence[str],
    simulator_id: str,
    work_root: str | Path,
    cache_dir: str | Path,
    history_end_day: float,
    observation_interval_days: float,
    oil_rate_relative_sigma: float,
    water_rate_relative_sigma: float,
    rate_sigma_floor: float,
    water_breakthrough_fraction: float = 0.05,
    producers: Sequence[str] = EGG_PRODUCERS,
    timeout: float = 600.0,
    min_free_disk_gb: float = 20.0,
) -> ForwardResult:
    """Run one restart-free Egg model and retain only compact observations."""

    field = np.asarray(logk, dtype=np.float64)
    if field.shape != EGG_SHAPE_ZYX or not np.all(np.isfinite(field)):
        raise ValueError(f"logk must be finite with shape {EGG_SHAPE_ZYX}")
    source = Path(template_dir).resolve()
    if not (source / "EGG.DATA").is_file():
        raise ValueError("template directory must contain EGG.DATA")
    if not simulator_command or not simulator_id:
        raise ValueError("simulator command and identity are required")
    observation_settings: dict[str, object] = {
        "producers": list(producers),
        "history_end_day": history_end_day,
        "observation_interval_days": observation_interval_days,
        "oil_rate_relative_sigma": oil_rate_relative_sigma,
        "water_rate_relative_sigma": water_rate_relative_sigma,
        "rate_sigma_floor": rate_sigma_floor,
        "water_breakthrough_fraction": water_breakthrough_fraction,
        "extractor_schema_version": 2,
    }

    def prepare(workdir: Path) -> None:
        shutil.copytree(source, workdir, dirs_exist_ok=True)
        write_egg_permeability_include(field, workdir / "mDARCY.INC")

    def extract(workdir: Path) -> dict[str, object]:
        return extract_egg_observations(
            workdir / "EGG",
            producers=producers,
            history_end_day=history_end_day,
            observation_interval_days=observation_interval_days,
            oil_rate_relative_sigma=oil_rate_relative_sigma,
            water_rate_relative_sigma=water_rate_relative_sigma,
            rate_sigma_floor=rate_sigma_floor,
            water_breakthrough_fraction=water_breakthrough_fraction,
        )

    cache_key = _egg_cache_key(
        field,
        template_dir=source,
        simulator_command=simulator_command,
        simulator_id=simulator_id,
        observation_settings=observation_settings,
    )
    return run_simulator(
        [*simulator_command, "EGG.DATA"],
        work_root=work_root,
        cache_dir=cache_dir,
        cache_key=cache_key,
        extractor=extract,
        prepare=prepare,
        timeout=timeout,
        min_free_disk_gb=min_free_disk_gb,
        disk_check_path=work_root,
    )
