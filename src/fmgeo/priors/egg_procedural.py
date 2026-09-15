"""A compact connected-channel procedural prior for the Egg geometry."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage


@dataclass(frozen=True)
class ProceduralEggResult:
    """One procedural facies realization and its permeability field."""

    facies: NDArray[np.uint8]
    logk: NDArray[np.float32]
    has_spanning_channel: bool


def _spans_x(facies: NDArray[np.uint8], active: NDArray[np.bool_]) -> bool:
    channel = (facies == 1) & active
    if not np.any(channel):
        return False
    active_x = np.flatnonzero(np.any(active, axis=(0, 1)))
    if active_x.size == 0:
        return False
    labels, count = ndimage.label(channel, structure=ndimage.generate_binary_structure(3, 1))
    left = set(np.unique(labels[..., active_x[0]][channel[..., active_x[0]]]))
    right = set(np.unique(labels[..., active_x[-1]][channel[..., active_x[-1]]]))
    return bool((left & right) - {0}) and count > 0


def generate_procedural_egg(
    active_mask: NDArray[np.bool_],
    *,
    seed: int,
    channel_width_cells: int,
    background_logk_mean: float = 4.6,
    channel_logk_mean: float = 7.6,
    logk_std: float = 0.25,
) -> ProceduralEggResult:
    """Generate a deterministic, vertically connected migrating channel belt."""

    active = np.asarray(active_mask, dtype=bool)
    if active.ndim != 3:
        raise ValueError("active_mask must have shape (z, y, x)")
    if not np.any(active):
        raise ValueError("active_mask must contain at least one active cell")
    if channel_width_cells < 1:
        raise ValueError("channel_width_cells must be positive")
    if logk_std < 0:
        raise ValueError("logk_std must be non-negative")

    nz, ny, nx = active.shape
    rng = np.random.default_rng(seed)
    half_width = channel_width_cells // 2
    low = min(max(half_width, 0), ny - 1)
    high = max(low + 1, ny - half_width)
    center = int(rng.integers(low, high))
    centerline = np.empty(nx, dtype=np.int64)
    centerline[0] = center
    for x_index in range(1, nx):
        step = int(rng.choice((-1, 0, 0, 0, 1)))
        centerline[x_index] = np.clip(centerline[x_index - 1] + step, low, high - 1)

    facies = np.zeros(active.shape, dtype=np.uint8)
    for z_index in range(nz):
        # Adjacent layer shifts differ by at most one cell, so a belt at least
        # three cells wide remains connected vertically.
        layer_shift = (z_index % 3) - 1
        for x_index, base_center in enumerate(centerline):
            layer_center = int(np.clip(base_center + layer_shift, 0, ny - 1))
            start = max(0, layer_center - half_width)
            stop = min(ny, start + channel_width_cells)
            start = max(0, stop - channel_width_cells)
            facies[z_index, start:stop, x_index] = 1

    facies[~active] = 255
    logk = rng.normal(background_logk_mean, logk_std, active.shape).astype(np.float32)
    channel_cells = (facies == 1) & active
    logk[channel_cells] = rng.normal(
        channel_logk_mean, logk_std, int(np.count_nonzero(channel_cells))
    ).astype(np.float32)
    logk[~active] = 0.0

    return ProceduralEggResult(
        facies=facies,
        logk=logk,
        has_spanning_channel=_spans_x(facies, active),
    )
