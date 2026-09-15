"""Deterministic conditional flow-matching training utilities."""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch import nn

SourceSampler = Callable[
    [int, torch.Generator, torch.device, torch.dtype], torch.Tensor
]


@dataclass(frozen=True)
class WhiteSourceSampler:
    """Seeded independent standard-normal source on a fixed spatial grid."""

    shape: tuple[int, int, int]

    def __post_init__(self) -> None:
        if any(size <= 0 for size in self.shape):
            raise ValueError("source shape entries must be positive")

    def __call__(
        self,
        batch_size: int,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.randn(
            (batch_size, 1, *self.shape),
            generator=generator,
            device=device,
            dtype=dtype,
        )


@dataclass(frozen=True)
class MaternSourceSampler:
    """Torch FFT sampler for the anisotropic Matérn FM source distribution."""

    shape: tuple[int, int, int]
    corr_len: tuple[float, float, float]
    nu: float = 1.5
    pad: float = 2.0

    def __post_init__(self) -> None:
        if any(size <= 0 for size in self.shape):
            raise ValueError("source shape entries must be positive")
        if any(length <= 0 for length in self.corr_len) or self.nu <= 0:
            raise ValueError("Matérn parameters must be positive")
        if self.pad < 1.5:
            raise ValueError("Matérn padding must be at least 1.5")

    def __call__(
        self,
        batch_size: int,
        generator: torch.Generator,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        padded_shape = tuple(math.ceil(self.pad * size) for size in self.shape)
        frequencies = [
            2.0 * torch.pi * torch.fft.fftfreq(size, device=device, dtype=dtype)
            for size in padded_shape
        ]
        grids = torch.meshgrid(*frequencies, indexing="ij")
        scaled = sum(
            (length * frequency) ** 2
            for length, frequency in zip(self.corr_len, grids, strict=True)
        )
        spectrum = (2.0 * self.nu + scaled) ** (-(self.nu + 1.5))
        spectrum = spectrum / spectrum.mean()
        white = torch.randn(
            (batch_size, 1, *padded_shape),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        filtered = torch.fft.ifftn(
            torch.fft.fftn(white, dim=(-3, -2, -1)) * torch.sqrt(spectrum),
            dim=(-3, -2, -1),
        ).real
        starts = tuple(
            (padded - size) // 2
            for padded, size in zip(padded_shape, self.shape, strict=True)
        )
        return filtered[
            ...,
            starts[0] : starts[0] + self.shape[0],
            starts[1] : starts[1] + self.shape[1],
            starts[2] : starts[2] + self.shape[2],
        ]


def make_source_sampler(
    kind: Literal["white", "matern"],
    *,
    shape: tuple[int, int, int],
    corr_len: tuple[float, float, float],
    nu: float,
    corr_len_scale: float = 1.0,
) -> WhiteSourceSampler | MaternSourceSampler:
    """Construct one ablation source while varying only its declared measure."""

    if corr_len_scale <= 0:
        raise ValueError("correlation-length scale must be positive")
    if kind == "white":
        return WhiteSourceSampler(shape=shape)
    if kind == "matern":
        scaled = (
            corr_len[0] * corr_len_scale,
            corr_len[1] * corr_len_scale,
            corr_len[2] * corr_len_scale,
        )
        return MaternSourceSampler(shape=shape, corr_len=scaled, nu=nu)
    raise ValueError(f"unsupported source kind {kind!r}")


def masked_flow_matching_loss(
    model: nn.Module,
    *,
    x0: torch.Tensor,
    x1: torch.Tensor,
    time: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Conditional FM squared error over active, non-padding cells only."""
    if x0.shape != x1.shape or x0.ndim != 5:
        raise ValueError("x0 and x1 must share shape (batch,channel,z,y,x)")
    if time.shape != (x0.shape[0],):
        raise ValueError("time must have shape (batch,)")
    try:
        active = torch.broadcast_to(mask.to(dtype=torch.bool, device=x0.device), x0.shape)
    except RuntimeError as error:
        raise ValueError("mask must broadcast to x0") from error
    if not torch.any(active):
        raise ValueError("mask contains no active cells")
    x0 = torch.where(active, x0, torch.zeros_like(x0))
    x1 = torch.where(active, x1, torch.zeros_like(x1))
    interpolated = (1.0 - time[:, None, None, None, None]) * x0
    interpolated = interpolated + time[:, None, None, None, None] * x1
    target = x1 - x0
    residual = model(interpolated, time) - target
    return torch.mean(residual[active] ** 2)


def seeded_batch_indices(
    length: int, *, batch_size: int, seed: int, epoch: int
) -> tuple[tuple[int, ...], ...]:
    """Return a repeatable, epoch-specific complete shuffled partition."""
    if length <= 0 or batch_size <= 0 or seed < 0 or epoch < 0:
        raise ValueError("length/batch size must be positive and seed/epoch non-negative")
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(length, generator=generator).tolist()
    return tuple(
        tuple(order[start : start + batch_size]) for start in range(0, length, batch_size)
    )


@dataclass(frozen=True)
class LayerTrendNormalizer:
    """Invertible per-layer mean and scale removal over active cells."""

    mean: torch.Tensor
    scale: torch.Tensor

    @classmethod
    def fit(cls, values: torch.Tensor, active_mask: torch.Tensor) -> LayerTrendNormalizer:
        if values.ndim != 5:
            raise ValueError("values must have shape (sample,channel,z,y,x)")
        active = torch.broadcast_to(active_mask.to(torch.bool), values.shape)
        means: list[torch.Tensor] = []
        scales: list[torch.Tensor] = []
        for z in range(values.shape[2]):
            selected = values[:, :, z][active[:, :, z]]
            if selected.numel() < 2:
                raise ValueError(f"layer {z} has insufficient active values")
            means.append(selected.mean())
            scales.append(selected.std().clamp_min(torch.finfo(values.dtype).eps))
        return cls(
            mean=torch.stack(means)[None, None, :, None, None],
            scale=torch.stack(scales)[None, None, :, None, None],
        )

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        return (values - self.mean.to(values)) / self.scale.to(values)

    def inverse(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.scale.to(values) + self.mean.to(values)


class ExponentialMovingAverage:
    """State-dict EMA used for all inference checkpoints."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        if not 0 < decay < 1:
            raise ValueError("EMA decay must lie strictly between zero and one")
        self.decay = decay
        self.shadow = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, value in model.state_dict().items():
            if value.is_floating_point():
                self.shadow[name].lerp_(value.detach(), 1.0 - self.decay)
            else:
                self.shadow[name].copy_(value.detach())

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self.shadow.items()}


def train_flow_matching(
    model: nn.Module,
    *,
    targets: torch.Tensor,
    source_sampler: SourceSampler,
    active_mask: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    batch_size: int,
    seed: int,
    ema_decay: float = 0.999,
    gradient_clip_norm: float = 1.0,
) -> tuple[list[float], ExponentialMovingAverage]:
    """Train deterministically from an in-memory normalized prior tensor."""
    if targets.ndim != 5 or epochs <= 0 or gradient_clip_norm <= 0:
        raise ValueError("targets must be 5D and training limits must be positive")
    device = next(model.parameters()).device
    targets = targets.to(device)
    mask = active_mask.to(device)
    ema = ExponentialMovingAverage(model, decay=ema_decay)
    history: list[float] = []
    for epoch in range(epochs):
        batches = seeded_batch_indices(
            len(targets), batch_size=batch_size, seed=seed, epoch=epoch
        )
        generator = torch.Generator(device=device).manual_seed(seed + 1_000_003 * epoch)
        for batch in batches:
            indices = torch.tensor(batch, device=device)
            x1 = targets.index_select(0, indices)
            x0 = source_sampler(len(batch), generator, device, targets.dtype)
            if x0.shape != x1.shape:
                raise ValueError("source sampler returned the wrong shape")
            time = torch.rand(len(batch), generator=generator, device=device, dtype=x1.dtype)
            optimizer.zero_grad(set_to_none=True)
            loss = masked_flow_matching_loss(
                model, x0=x0, x1=x1, time=time, mask=mask
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            ema.update(model)
            history.append(float(loss.detach()))
    return history, ema


def _atomic_torch_save(payload: dict[str, Any], destination: Path) -> None:
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def save_checkpoint_policy(
    directory: str | Path,
    *,
    ema_state: dict[str, Any],
    resume_state: dict[str, Any],
) -> None:
    """Atomically retain only EMA inference weights and the latest resume state."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    _atomic_torch_save(ema_state, root / "ema.pt")
    _atomic_torch_save(resume_state, root / "resume.pt")
    for pattern in ("ema-step-*.pt", "resume-step-*.pt"):
        for obsolete in root.glob(pattern):
            obsolete.unlink()
