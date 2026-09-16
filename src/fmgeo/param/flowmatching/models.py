"""Strict velocity-model construction shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from torch import nn

from fmgeo.param.flowmatching.model_unet3d import UNet3D
from fmgeo.param.flowmatching.model_uno3d import UNO3D


def _integer(config: Mapping[str, object], name: str) -> int:
    value = config.get(name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"model {name} must be an integer")
    return value


def build_velocity_model(config: Mapping[str, object]) -> nn.Module:
    """Build a U-Net or UNO only from its complete declared configuration."""
    kind = config.get("kind")
    if kind == "unet3d":
        expected = {
            "kind",
            "in_channels",
            "base_channels",
            "time_dim",
            "coarse_attention_only",
        }
        if set(config) != expected or config.get("coarse_attention_only") is not True:
            raise ValueError("invalid or incomplete unet3d model configuration")
        return UNet3D(
            in_channels=_integer(config, "in_channels"),
            base_channels=_integer(config, "base_channels"),
            time_dim=_integer(config, "time_dim"),
        )
    if kind == "uno3d":
        expected = {
            "kind",
            "in_channels",
            "hidden_channels",
            "time_dim",
            "modes_zyx",
            "blocks",
        }
        modes = config.get("modes_zyx")
        if set(config) != expected or not isinstance(modes, Sequence) or len(modes) != 3:
            raise ValueError("invalid or incomplete uno3d model configuration")
        parsed_modes = tuple(int(value) for value in modes)
        return UNO3D(
            in_channels=_integer(config, "in_channels"),
            hidden_channels=_integer(config, "hidden_channels"),
            time_dim=_integer(config, "time_dim"),
            modes=(parsed_modes[0], parsed_modes[1], parsed_modes[2]),
            blocks=_integer(config, "blocks"),
        )
    raise ValueError(f"unsupported velocity model kind {kind!r}")
