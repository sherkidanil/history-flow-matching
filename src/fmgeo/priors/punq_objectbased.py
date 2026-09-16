"""Conditioned object-based geological prior for the PUNQ-S3 grid."""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import h5py  # type: ignore[import-untyped]
import numpy as np
import yaml
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy.stats import wasserstein_distance

from fmgeo.matern import matern_field
from fmgeo.metrics.geology import experimental_variogram


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FaciesProperty(_StrictModel):
    """Log-permeability distribution and deterministic porosity transform."""

    logk_mean: float
    logk_std: float = Field(gt=0)
    porosity_intercept: float
    porosity_slope: float
    porosity_bounds: tuple[float, float]

    @model_validator(mode="after")
    def validate_bounds(self) -> FaciesProperty:
        lower, upper = self.porosity_bounds
        if not 0 <= lower < upper <= 1:
            raise ValueError("porosity bounds must satisfy 0 <= lower < upper <= 1")
        return self


class ChannelGeometry(_StrictModel):
    """Distributions for sinusoidal channel centerlines, in physical units."""

    width_mean_m: float = Field(gt=0)
    width_std_m: float = Field(ge=0)
    amplitude_range_m: tuple[float, float]
    wavelength_range_m: tuple[float, float]
    azimuth_range_degrees: tuple[float, float]
    net_to_gross_width_fraction: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def validate_ranges(self) -> ChannelGeometry:
        for name in (
            "amplitude_range_m",
            "wavelength_range_m",
            "azimuth_range_degrees",
        ):
            lower, upper = getattr(self, name)
            if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
                raise ValueError(f"{name} must be a finite ordered range")
        if self.amplitude_range_m[0] < 0 or self.wavelength_range_m[0] <= 0:
            raise ValueError("amplitudes must be non-negative and wavelengths positive")
        return self


class LayerPrior(_StrictModel):
    """One layer's geometry, properties, and permeability anisotropy."""

    kind: Literal["channel", "homogeneous"]
    background: FaciesProperty
    sand: FaciesProperty | None = None
    geometry: ChannelGeometry | None = None
    vertical_ratio: float = Field(gt=0)
    property_corr_len_cells: tuple[float, float]

    @model_validator(mode="after")
    def validate_layer(self) -> LayerPrior:
        if any(value <= 0 for value in self.property_corr_len_cells):
            raise ValueError("property correlation lengths must be positive")
        if self.kind == "channel" and (self.sand is None or self.geometry is None):
            raise ValueError("channel layers require sand properties and geometry")
        if self.kind == "homogeneous" and (
            self.sand is not None or self.geometry is not None
        ):
            raise ValueError("homogeneous layers cannot define sand or channel geometry")
        return self


class HardDatum(_StrictModel):
    """Known facies label at one zero-based grid cell."""

    z: int = Field(ge=0)
    y: int = Field(ge=0)
    x: int = Field(ge=0)
    facies: Literal[0, 1]


class PunqPriorConfig(_StrictModel):
    """Complete, validated object-prior configuration."""

    seed: int = Field(default=0, ge=0)
    shape: tuple[int, int, int]
    cell_size_m: tuple[float, float]
    channels_per_channel_layer: int = Field(gt=0)
    layers: tuple[LayerPrior, ...]
    hard_data: tuple[HardDatum, ...]

    @model_validator(mode="after")
    def validate_grid(self) -> PunqPriorConfig:
        if any(size <= 0 for size in self.shape):
            raise ValueError("shape entries must be positive")
        if any(size <= 0 for size in self.cell_size_m):
            raise ValueError("cell sizes must be positive")
        if len(self.layers) != self.shape[0]:
            raise ValueError("one layer configuration is required per z layer")
        locations: set[tuple[int, int, int]] = set()
        for datum in self.hard_data:
            location = (datum.z, datum.y, datum.x)
            if location in locations:
                raise ValueError(f"duplicate hard-data location: {location}")
            if any(index >= size for index, size in zip(location, self.shape, strict=True)):
                raise ValueError(f"hard-data location outside grid: {location}")
            locations.add(location)
        return self


@dataclass(frozen=True)
class Channel:
    """One sampled sinusoidal centerline in projected coordinates."""

    azimuth_degrees: float
    amplitude_m: float
    wavelength_m: float
    phase_radians: float
    offset_m: float
    width_m: float


@dataclass(frozen=True)
class PUNQRealization:
    """Generated facies, rock properties, and sampled geological objects."""

    facies: NDArray[np.uint8]
    logk: NDArray[np.float32]
    porosity: NDArray[np.float32]
    permx: NDArray[np.float32]
    permy: NDArray[np.float32]
    permz: NDArray[np.float32]
    channels: dict[int, tuple[Channel, ...]]


def load_punq_prior_config(path: str | Path) -> PunqPriorConfig:
    """Load a strict YAML prior configuration."""
    with Path(path).open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ValueError("PUNQ prior configuration root must be a mapping")
    return PunqPriorConfig.model_validate(payload)


def _projected_coordinates(
    ny: int, nx: int, cell_size_m: tuple[float, float], azimuth_degrees: float
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    dy, dx = cell_size_m
    y, x = np.meshgrid(
        (np.arange(ny, dtype=np.float64) + 0.5) * dy,
        (np.arange(nx, dtype=np.float64) + 0.5) * dx,
        indexing="ij",
    )
    angle = np.deg2rad(azimuth_degrees)
    along = x * np.cos(angle) + y * np.sin(angle)
    across = -x * np.sin(angle) + y * np.cos(angle)
    return along, across


def _sample_channels(
    *,
    layer_index: int,
    config: PunqPriorConfig,
    rng: np.random.Generator,
) -> tuple[Channel, ...]:
    layer = config.layers[layer_index]
    geometry = layer.geometry
    if geometry is None:
        return ()
    ny, nx = config.shape[1:]
    sand_targets = [
        datum
        for datum in config.hard_data
        if datum.z == layer_index and datum.facies == 1
    ]
    channels: list[Channel] = []
    for index in range(config.channels_per_channel_layer):
        azimuth = rng.uniform(*geometry.azimuth_range_degrees)
        amplitude = rng.uniform(*geometry.amplitude_range_m)
        wavelength = rng.uniform(*geometry.wavelength_range_m)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        gross_width = max(
            min(config.cell_size_m) * 0.5,
            rng.normal(geometry.width_mean_m, geometry.width_std_m),
        )
        width = gross_width * geometry.net_to_gross_width_fraction
        along, across = _projected_coordinates(ny, nx, config.cell_size_m, azimuth)
        if sand_targets:
            target = sand_targets[index % len(sand_targets)]
            offset = across[target.y, target.x] - amplitude * np.sin(
                2.0 * np.pi * along[target.y, target.x] / wavelength + phase
            )
        else:
            offset = rng.uniform(float(across.min()), float(across.max()))
        channels.append(
            Channel(
                azimuth_degrees=float(azimuth),
                amplitude_m=float(amplitude),
                wavelength_m=float(wavelength),
                phase_radians=float(phase),
                offset_m=float(offset),
                width_m=float(width),
            )
        )
    return tuple(channels)


def _rasterize_channels(
    channels: tuple[Channel, ...],
    *,
    shape: tuple[int, int],
    cell_size_m: tuple[float, float],
) -> NDArray[np.bool_]:
    facies = np.zeros(shape, dtype=bool)
    for channel in channels:
        along, across = _projected_coordinates(
            shape[0], shape[1], cell_size_m, channel.azimuth_degrees
        )
        centerline = channel.offset_m + channel.amplitude_m * np.sin(
            2.0 * np.pi * along / channel.wavelength_m + channel.phase_radians
        )
        facies |= np.abs(across - centerline) <= channel.width_m / 2.0
    return facies


def _standardized_property_field(
    shape: tuple[int, int],
    corr_len: tuple[float, float],
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    field = matern_field(
        (1, *shape),
        (0.5, *corr_len),
        rng=rng,
        pad=2.0,
    )[0]
    standard_deviation = float(field.std())
    if standard_deviation <= np.finfo(np.float64).eps:
        raise RuntimeError("sampled property field has zero variance")
    return (field - field.mean()) / standard_deviation


def generate_punq_realization(
    config: PunqPriorConfig,
    active_mask: NDArray[np.bool_],
    *,
    seed: int,
) -> PUNQRealization:
    """Generate one deterministic conditioned prior realization."""
    active = np.asarray(active_mask, dtype=bool)
    if active.shape != config.shape:
        raise ValueError(f"active mask must have shape {config.shape}")
    for datum in config.hard_data:
        if not active[datum.z, datum.y, datum.x]:
            raise ValueError(
                f"hard-data location is inactive: {(datum.z, datum.y, datum.x)}"
            )

    rng = np.random.default_rng(seed)
    facies = np.zeros(config.shape, dtype=np.uint8)
    logk = np.zeros(config.shape, dtype=np.float64)
    porosity = np.zeros(config.shape, dtype=np.float64)
    channels_by_layer: dict[int, tuple[Channel, ...]] = {}

    for z, layer in enumerate(config.layers):
        if layer.kind == "channel":
            channels = _sample_channels(layer_index=z, config=config, rng=rng)
            channels_by_layer[z] = channels
            facies[z] = _rasterize_channels(
                channels,
                shape=config.shape[1:],
                cell_size_m=config.cell_size_m,
            )
        for datum in config.hard_data:
            if datum.z == z:
                facies[z, datum.y, datum.x] = datum.facies

        correlated = _standardized_property_field(
            config.shape[1:], layer.property_corr_len_cells, rng
        )
        for label, properties in ((0, layer.background), (1, layer.sand)):
            if properties is None:
                continue
            selected = (facies[z] == label) & active[z]
            logk[z, selected] = (
                properties.logk_mean + properties.logk_std * correlated[selected]
            )
            porosity[z, selected] = np.clip(
                properties.porosity_intercept
                + properties.porosity_slope * logk[z, selected],
                *properties.porosity_bounds,
            )

    facies[~active] = 255
    permx = np.exp(logk)
    permx[~active] = 0.0
    permy = permx.copy()
    permz = np.empty_like(permx)
    for z, layer in enumerate(config.layers):
        permz[z] = permx[z] * layer.vertical_ratio
    permz[~active] = 0.0
    return PUNQRealization(
        facies=facies,
        logk=logk.astype(np.float32),
        porosity=porosity.astype(np.float32),
        permx=permx.astype(np.float32),
        permy=permy.astype(np.float32),
        permz=permz.astype(np.float32),
        channels=channels_by_layer,
    )


def _sample_seed(base_seed: int, sample_index: int) -> int:
    state = np.random.SeedSequence([base_seed, sample_index]).generate_state(
        1, dtype=np.uint64
    )
    return int(state[0])


def write_punq_prior_hdf5(
    path: str | Path,
    *,
    config: PunqPriorConfig,
    active_mask: NDArray[np.bool_],
    count: int,
    seed: int,
) -> dict[str, object]:
    """Atomically stream the required prior tensors to a compact HDF5 artifact."""
    if count <= 0:
        raise ValueError("count must be positive")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        with h5py.File(temporary_path, "w") as handle:
            chunks = (min(count, 64), *config.shape)
            logk_dataset = handle.create_dataset(
                "logk",
                shape=(count, *config.shape),
                dtype="float32",
                chunks=chunks,
                compression="gzip",
                shuffle=True,
            )
            facies_dataset = handle.create_dataset(
                "facies",
                shape=(count, *config.shape),
                dtype="uint8",
                chunks=chunks,
                compression="gzip",
                shuffle=True,
            )
            handle.create_dataset(
                "active_mask", data=np.asarray(active_mask, dtype=np.uint8), dtype="uint8"
            )
            seed_dataset = handle.create_dataset("sample_seeds", shape=(count,), dtype="uint64")
            handle.attrs["schema_version"] = 1
            handle.attrs["config"] = config.model_dump_json()
            handle.attrs["base_seed"] = seed
            for index in range(count):
                sample_seed = _sample_seed(seed, index)
                realization = generate_punq_realization(
                    config, active_mask, seed=sample_seed
                )
                logk_dataset[index] = realization.logk
                facies_dataset[index] = realization.facies
                seed_dataset[index] = sample_seed
            handle.flush()
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return {
        "path": str(destination),
        "shape": (count, *config.shape),
        "dtype": "float32",
        "facies_dtype": "uint8",
        "count": count,
        "seed": seed,
    }


def validate_prior_against_truth(
    *,
    config: PunqPriorConfig,
    active_mask: NDArray[np.bool_],
    facies_ensemble: NDArray[np.uint8],
    logk_ensemble: NDArray[np.floating],
    truth_logk: NDArray[np.floating],
    truth_porosity: NDArray[np.floating],
) -> dict[str, object]:
    """Compare reduced-ensemble facies, histogram, and variograms with truth."""
    active = np.asarray(active_mask, dtype=bool)
    facies = np.asarray(facies_ensemble, dtype=np.uint8)
    logk = np.asarray(logk_ensemble, dtype=np.float64)
    truth_k = np.asarray(truth_logk, dtype=np.float64)
    truth_phi = np.asarray(truth_porosity, dtype=np.float64)
    expected_ensemble_shape = (facies.shape[0], *config.shape)
    if (
        facies.shape != expected_ensemble_shape
        or logk.shape != expected_ensemble_shape
        or active.shape != config.shape
        or truth_k.shape != config.shape
        or truth_phi.shape != config.shape
    ):
        raise ValueError("prior validation arrays do not match the configured grid")
    if facies.shape[0] < 2:
        raise ValueError("prior validation requires at least two realizations")

    channel_layers = [
        index for index, layer in enumerate(config.layers) if layer.kind == "channel"
    ]
    fraction_details: dict[str, dict[str, float | bool]] = {}
    fractions_accepted = True
    for z in channel_layers:
        active_count = int(active[z].sum())
        sample_fractions = ((facies[:, z] == 1) & active[z]).sum(axis=(1, 2)) / active_count
        truth_fraction = float(((truth_phi[z] >= 0.20) & active[z]).sum() / active_count)
        lower, upper = np.quantile(sample_fractions, [0.025, 0.975])
        covered = bool(lower <= truth_fraction <= upper)
        fractions_accepted &= covered
        fraction_details[str(z + 1)] = {
            "truth": truth_fraction,
            "prior_q025": float(lower),
            "prior_q975": float(upper),
            "covered": covered,
        }

    truth_active = truth_k[active]
    prior_active = logk[:, active].ravel()
    truth_scale = float(truth_active.std())
    if truth_scale <= np.finfo(np.float64).eps:
        raise ValueError("truth log-permeability has zero variance")
    normalized_wasserstein = float(
        wasserstein_distance(prior_active, truth_active) / truth_scale
    )

    covered_lags = 0
    total_lags = 0
    variogram_details: dict[str, dict[str, object]] = {}
    for axis, axis_name in enumerate(("z", "y", "x")):
        max_lag = min(4, config.shape[axis] - 1)
        sample_curves = np.stack(
            [
                experimental_variogram(
                    sample, axis=axis, max_lag=max_lag, active_mask=active
                )
                for sample in logk
            ]
        )
        truth_curve = experimental_variogram(
            truth_k, axis=axis, max_lag=max_lag, active_mask=active
        )
        lower = np.nanquantile(sample_curves, 0.025, axis=0)
        upper = np.nanquantile(sample_curves, 0.975, axis=0)
        valid = np.isfinite(truth_curve[1:]) & np.isfinite(lower[1:]) & np.isfinite(
            upper[1:]
        )
        covered = (truth_curve[1:] >= lower[1:]) & (truth_curve[1:] <= upper[1:]) & valid
        covered_lags += int(covered.sum())
        total_lags += int(valid.sum())
        variogram_details[axis_name] = {
            "truth": truth_curve.tolist(),
            "prior_q025": lower.tolist(),
            "prior_q975": upper.tolist(),
            "covered": covered.tolist(),
        }
    variogram_coverage = covered_lags / total_lags if total_lags else 0.0
    accepted = (
        fractions_accepted
        and normalized_wasserstein <= 0.5
        and variogram_coverage >= 0.70
    )
    return {
        "accepted": accepted,
        "sand_fraction_by_layer": fraction_details,
        "histogram_normalized_wasserstein": normalized_wasserstein,
        "histogram_threshold": 0.5,
        "variogram_envelope_coverage": variogram_coverage,
        "variogram_coverage_threshold": 0.70,
        "variograms": variogram_details,
        "truth_sand_definition": "porosity >= 0.20 in channel layers",
    }
