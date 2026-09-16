"""Reusable Egg flow-matching checkpoint adapter for inversion diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from m8_train_egg import load_fm_config

from fmgeo.artifacts import canonical_config_hash, sha256_file
from fmgeo.param.flowmatching.models import build_velocity_model
from fmgeo.param.flowmatching.sample import FlowTransform
from fmgeo.param.flowmatching.train import LayerTrendNormalizer
from fmgeo.runtime import select_device


def embed_active(parameters: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Embed active-cell parameter vectors into full Egg fields."""

    values = np.asarray(parameters, dtype=np.float64)
    mask = np.asarray(active, dtype=bool)
    if values.ndim != 2 or mask.ndim != 3 or values.shape[1] != int(mask.sum()):
        raise ValueError("parameters must match the number of active grid cells")
    fields = np.zeros((len(values), *mask.shape), dtype=np.float64)
    fields[:, mask] = values
    return fields


def build_flow_adapter(
    initial_fields: np.ndarray,
    active: np.ndarray,
    *,
    checkpoint_path: Path,
    training_data: Path,
    fm_config_path: Path,
    strategy: str,
    device_name: Literal["auto", "cpu", "mps", "cuda"],
    batch_size: int,
) -> tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], dict[str, object]]:
    """Load a validated checkpoint and expose active latent encode/decode arrays."""

    if batch_size < 1:
        raise ValueError("batch size must be positive")
    config = load_fm_config(fm_config_path)
    config_hash = canonical_config_hash(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if checkpoint["strategy"] != strategy or checkpoint["config_hash"] != config_hash:
        raise ValueError("FM checkpoint strategy or configuration does not match")
    if checkpoint["data_sha256"] != sha256_file(training_data):
        raise ValueError("FM checkpoint training-data hash does not match")
    selected = select_device(
        device_name,
        cuda_available=torch.cuda.is_available(),
        mps_available=bool(torch.backends.mps.is_available()),
    )
    device = torch.device(selected)
    model = build_velocity_model(config.model.model_dump(mode="python")).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    mask = torch.from_numpy(np.asarray(active, dtype=bool))[None, None].to(device)
    normalizer = LayerTrendNormalizer(
        mean=checkpoint["normalizer_mean"], scale=checkpoint["normalizer_scale"]
    )
    transform = FlowTransform(
        model, steps=config.integration.steps, method=config.integration.method, mask=mask
    )

    def apply(values: np.ndarray, *, inverse: bool) -> np.ndarray:
        batches: list[np.ndarray] = []
        for start in range(0, len(values), batch_size):
            tensor = torch.from_numpy(values[start : start + batch_size].astype(np.float32))
            tensor = tensor[:, None].to(device)
            if inverse:
                tensor = normalizer.transform(tensor)
                transformed = transform.inverse(tensor)
            else:
                transformed = transform.forward(tensor)
                transformed = normalizer.inverse(transformed)
            transformed = torch.where(mask, transformed, torch.zeros_like(transformed))
            batches.append(transformed[:, 0].cpu().numpy().astype(np.float64))
        return np.concatenate(batches)

    latent_fields = apply(initial_fields, inverse=True)
    initial_parameters = latent_fields[:, np.asarray(active, dtype=bool)]

    def decode(parameters: np.ndarray) -> np.ndarray:
        return apply(embed_active(parameters, active), inverse=False)

    metadata: dict[str, object] = {
        "device": selected,
        "fm_config_hash": config_hash,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "training_data_sha256": sha256_file(training_data),
    }
    return initial_parameters, decode, metadata
