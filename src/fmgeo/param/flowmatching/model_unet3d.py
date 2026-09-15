"""Shape-preserving 3D U-Net velocity model with sinusoidal time FiLM."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalTimeEmbedding(nn.Module):
    """Continuous transformer-style embedding for scalar ODE time."""

    def __init__(self, dimension: int) -> None:
        super().__init__()
        if dimension < 4 or dimension % 2:
            raise ValueError("time embedding dimension must be even and at least four")
        self.dimension = dimension

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        if time.ndim != 1:
            raise ValueError("time must have shape (batch,)")
        half = self.dimension // 2
        scale = math.log(10_000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=time.device, dtype=time.dtype)
        )
        angles = time[:, None] * frequencies[None, :]
        return torch.cat((angles.sin(), angles.cos()), dim=1)


def pad_to_multiple(
    values: torch.Tensor, multiple: int = 8
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Reflection-pad spatial dimensions on the high side and return crop sizes."""
    if values.ndim != 5 or multiple <= 0:
        raise ValueError("values must have shape (batch,channel,z,y,x)")
    original = (
        int(values.shape[-3]),
        int(values.shape[-2]),
        int(values.shape[-1]),
    )
    padding_zyx = tuple((-size) % multiple for size in original)
    padding = (0, padding_zyx[2], 0, padding_zyx[1], 0, padding_zyx[0])
    if not any(padding_zyx):
        return values, original
    mode = (
        "reflect"
        if all(pad < size for pad, size in zip(padding_zyx, original, strict=True))
        else "replicate"
    )
    return F.pad(values, padding, mode=mode), original


def crop_to_shape(values: torch.Tensor, shape: tuple[int, int, int]) -> torch.Tensor:
    return values[..., : shape[0], : shape[1], : shape[2]]


def _groups(channels: int) -> int:
    return math.gcd(channels, 4)


class TimeResidualBlock(nn.Module):
    """Residual 3D convolution block modulated by a time embedding."""

    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.film = nn.Linear(time_dim, 2 * out_channels)
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv3d(in_channels, out_channels, 1)
        )

    def forward(self, values: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(values)))
        scale, shift = self.film(time_embedding).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None, None])
        hidden = hidden + shift[:, :, None, None, None]
        hidden = self.conv2(F.silu(hidden))
        return hidden + self.skip(values)


class SelfAttention3D(nn.Module):
    """Self-attention used only at coarse U-Net resolutions."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, num_heads=1, batch_first=True)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, channels, z, y, x = values.shape
        tokens = values.reshape(batch, channels, z * y * x).transpose(1, 2)
        normalized = self.norm(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        return (tokens + attended).transpose(1, 2).reshape(batch, channels, z, y, x)


class UNet3D(nn.Module):
    """Compact 3D U-Net for conditional flow-matching velocity prediction."""

    def __init__(self, in_channels: int, base_channels: int = 16, time_dim: int = 64) -> None:
        super().__init__()
        if in_channels <= 0 or base_channels <= 0:
            raise ValueError("channel counts must be positive")
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input = nn.Conv3d(in_channels, base_channels, 3, padding=1)
        self.encoder1 = TimeResidualBlock(base_channels, base_channels, time_dim)
        self.down1 = nn.Conv3d(base_channels, 2 * base_channels, 4, stride=2, padding=1)
        self.encoder2 = TimeResidualBlock(2 * base_channels, 2 * base_channels, time_dim)
        self.attention2 = SelfAttention3D(2 * base_channels)
        self.down2 = nn.Conv3d(2 * base_channels, 4 * base_channels, 4, stride=2, padding=1)
        self.middle = TimeResidualBlock(4 * base_channels, 4 * base_channels, time_dim)
        self.attention_middle = SelfAttention3D(4 * base_channels)
        self.up2 = nn.ConvTranspose3d(
            4 * base_channels, 2 * base_channels, 4, stride=2, padding=1
        )
        self.decoder2 = TimeResidualBlock(4 * base_channels, 2 * base_channels, time_dim)
        self.up1 = nn.ConvTranspose3d(
            2 * base_channels, base_channels, 4, stride=2, padding=1
        )
        self.decoder1 = TimeResidualBlock(2 * base_channels, base_channels, time_dim)
        self.output = nn.Conv3d(base_channels, in_channels, 3, padding=1)

    def forward(self, values: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        padded, original_shape = pad_to_multiple(values, 8)
        embedded = self.time(time)
        skip1 = self.encoder1(self.input(padded), embedded)
        skip2 = self.attention2(self.encoder2(self.down1(skip1), embedded))
        hidden = self.attention_middle(self.middle(self.down2(skip2), embedded))
        hidden = self.decoder2(torch.cat((self.up2(hidden), skip2), dim=1), embedded)
        hidden = self.decoder1(torch.cat((self.up1(hidden), skip1), dim=1), embedded)
        return crop_to_shape(self.output(hidden), original_shape)
