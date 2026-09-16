"""Resolution-flexible 3D neural operator velocity model."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from fmgeo.param.flowmatching.model_unet3d import (
    SinusoidalTimeEmbedding,
    crop_to_shape,
    pad_to_multiple,
)


class SpectralConv3D(nn.Module):
    """Low-mode Fourier convolution over three spatial dimensions."""

    def __init__(
        self, in_channels: int, out_channels: int, modes: tuple[int, int, int]
    ) -> None:
        super().__init__()
        if any(mode <= 0 for mode in modes):
            raise ValueError("spectral mode counts must be positive")
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / (in_channels * out_channels) ** 0.5
        weights = scale * torch.randn(in_channels, out_channels, *modes, dtype=torch.cfloat)
        self.weights = nn.Parameter(weights)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        transformed = torch.fft.rfftn(values, dim=(-3, -2, -1), norm="ortho")
        transformed = torch.fft.fftshift(transformed, dim=(-3, -2))
        batch, _, z, y, x_frequency = transformed.shape
        mz = min(self.modes[0], z)
        my = min(self.modes[1], y)
        mx = min(self.modes[2], x_frequency)
        z_start = (z - mz) // 2
        y_start = (y - my) // 2
        output = torch.zeros(
            batch,
            self.out_channels,
            z,
            y,
            x_frequency,
            dtype=transformed.dtype,
            device=values.device,
        )
        selected = transformed[:, :, z_start : z_start + mz, y_start : y_start + my, :mx]
        output[:, :, z_start : z_start + mz, y_start : y_start + my, :mx] = torch.einsum(
            "bizyx,iozyx->bozyx", selected, self.weights[:, :, :mz, :my, :mx]
        )
        output = torch.fft.ifftshift(output, dim=(-3, -2))
        return torch.fft.irfftn(output, s=values.shape[-3:], dim=(-3, -2, -1), norm="ortho")


class OperatorBlock(nn.Module):
    def __init__(self, channels: int, modes: tuple[int, int, int]) -> None:
        super().__init__()
        self.spectral = SpectralConv3D(channels, channels, modes)
        self.local = nn.Conv3d(channels, channels, 1)
        self.norm = nn.GroupNorm(1, channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.norm(self.spectral(values) + self.local(values)))


class UNO3D(nn.Module):
    """Small Fourier neural operator that accepts arbitrary spatial resolution."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 24,
        time_dim: int = 64,
        modes: tuple[int, int, int] = (4, 8, 8),
        blocks: int = 3,
    ) -> None:
        super().__init__()
        if blocks <= 0:
            raise ValueError("UNO3D needs at least one operator block")
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, 2 * hidden_channels)
        )
        self.lift = nn.Conv3d(in_channels, hidden_channels, 1)
        self.blocks = nn.ModuleList(
            OperatorBlock(hidden_channels, modes) for _ in range(blocks)
        )
        self.project = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels, 1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, in_channels, 1),
        )

    def forward(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        padded, original_shape = pad_to_multiple(values, 8)
        hidden = self.lift(padded)
        scale, shift = self.time(time).chunk(2, dim=1)
        hidden = hidden * (1.0 + scale[:, :, None, None, None])
        hidden = hidden + shift[:, :, None, None, None]
        for block in self.blocks:
            hidden = block(hidden)
        return crop_to_shape(self.project(hidden), original_shape)
