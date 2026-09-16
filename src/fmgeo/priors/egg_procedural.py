"""A compact connected-channel procedural prior for the Egg geometry."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage

_FACE_NEIGHBORS = (
    (1, 0, 0),
    (-1, 0, 0),
    (0, 1, 0),
    (0, -1, 0),
    (0, 0, 1),
    (0, 0, -1),
)


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


@lru_cache(maxsize=8)
def _distance_to_right(shape: tuple[int, int, int], active_bytes: bytes) -> NDArray[np.int32]:
    """Cache the expensive geodesic distance map for a fixed active mask."""

    active = np.frombuffer(active_bytes, dtype=np.bool_).reshape(shape)
    nz, ny, nx = shape
    active_x = np.flatnonzero(np.any(active, axis=(0, 1)))
    if active_x.size == 0:
        raise ValueError("active_mask must contain at least one active cell")
    right_x = int(active_x[-1])

    distance = np.full(active.shape, -1, dtype=np.int32)
    queue: deque[tuple[int, int, int]] = deque()
    for z_index, y_index in np.argwhere(active[..., right_x]):
        cell = (int(z_index), int(y_index), right_x)
        distance[cell] = 0
        queue.append(cell)

    while queue:
        z_index, y_index, x_index = queue.popleft()
        next_distance = distance[z_index, y_index, x_index] + 1
        for dz, dy, dx in _FACE_NEIGHBORS:
            neighbor = (z_index + dz, y_index + dy, x_index + dx)
            if not (0 <= neighbor[0] < nz and 0 <= neighbor[1] < ny and 0 <= neighbor[2] < nx):
                continue
            if active[neighbor] and distance[neighbor] < 0:
                distance[neighbor] = next_distance
                queue.append(neighbor)

    distance.flags.writeable = False
    return distance


def _random_spanning_path(
    active: NDArray[np.bool_], rng: np.random.Generator
) -> NDArray[np.bool_]:
    """Trace a random shortest active-cell path between the x boundaries."""

    nz, ny, nx = active.shape
    distance = _distance_to_right(active.shape, active.tobytes())
    active_x = np.flatnonzero(np.any(active, axis=(0, 1)))
    left_x = int(active_x[0])

    starts = np.argwhere((distance[..., left_x] >= 0) & active[..., left_x])
    if starts.size == 0:
        raise ValueError("active_mask has no connected path between its x boundaries")
    start_z, start_y = starts[int(rng.integers(len(starts)))]
    current = (int(start_z), int(start_y), left_x)
    path = np.zeros(active.shape, dtype=bool)
    path[current] = True
    while distance[current] > 0:
        target_distance = distance[current] - 1
        candidates: list[tuple[int, int, int]] = []
        for dz, dy, dx in _FACE_NEIGHBORS:
            neighbor = (current[0] + dz, current[1] + dy, current[2] + dx)
            if not (0 <= neighbor[0] < nz and 0 <= neighbor[1] < ny and 0 <= neighbor[2] < nx):
                continue
            if distance[neighbor] == target_distance:
                candidates.append(neighbor)
        current = candidates[int(rng.integers(len(candidates)))]
        path[current] = True
    return path


def generate_procedural_egg(
    active_mask: NDArray[np.bool_],
    *,
    seed: int,
    channel_width_cells: int,
    channel_count: int = 1,
    background_logk_mean: float = 4.6,
    channel_logk_mean: float = 7.6,
    background_logk_std: float = 0.25,
    channel_logk_std: float = 0.25,
) -> ProceduralEggResult:
    """Generate a deterministic, vertically connected migrating channel belt."""

    active = np.asarray(active_mask, dtype=bool)
    if active.ndim != 3:
        raise ValueError("active_mask must have shape (z, y, x)")
    if not np.any(active):
        raise ValueError("active_mask must contain at least one active cell")
    if channel_width_cells < 1:
        raise ValueError("channel_width_cells must be positive")
    if channel_count < 1:
        raise ValueError("channel_count must be positive")
    if background_logk_std < 0 or channel_logk_std < 0:
        raise ValueError("log-k standard deviations must be non-negative")

    rng = np.random.default_rng(seed)
    path = np.zeros(active.shape, dtype=bool)
    for _ in range(channel_count):
        path |= _random_spanning_path(active, rng)
    dilation_steps = max(0, (channel_width_cells - 1) // 2)
    channel = (
        ndimage.binary_dilation(
            path,
            structure=ndimage.generate_binary_structure(3, 1),
            iterations=dilation_steps,
        )
        if dilation_steps
        else path
    )
    channel &= active
    facies = np.zeros(active.shape, dtype=np.uint8)
    facies[channel] = 1

    facies[~active] = 255
    logk = rng.normal(background_logk_mean, background_logk_std, active.shape).astype(
        np.float32
    )
    channel_cells = (facies == 1) & active
    logk[channel_cells] = rng.normal(
        channel_logk_mean, channel_logk_std, int(np.count_nonzero(channel_cells))
    ).astype(np.float32)
    logk[~active] = 0.0

    return ProceduralEggResult(
        facies=facies,
        logk=logk,
        has_spanning_channel=_spans_x(facies, active),
    )
