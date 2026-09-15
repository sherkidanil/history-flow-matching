"""Simulator-facing serialization for Egg permeability fields."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

EGG_SHAPE_ZYX = (7, 60, 60)


def write_egg_permeability_include(logk: ArrayLike, path: str | Path) -> None:
    """Write PERMX and the benchmark's standard PERMY/PERMZ relationships."""

    values = np.asarray(logk, dtype=np.float64)
    if values.shape != EGG_SHAPE_ZYX or not np.all(np.isfinite(values)):
        raise ValueError(f"logk must be a finite field with shape {EGG_SHAPE_ZYX}")
    permeability = np.exp(values)
    if not np.all(np.isfinite(permeability)) or np.any(permeability <= 0):
        raise ValueError("exp(logk) must be finite and positive")
    flattened = permeability.ravel()
    lines = [
        " ".join(f"{value:.8g}" for value in flattened[start : start + 8])
        for start in range(0, len(flattened), 8)
    ]
    content = (
        "PERMX\n"
        + "\n".join(lines)
        + "\n/\n\nCOPY\n"
        + " 'PERMX' 'PERMY'  1 60 1 60 1 7 /\n"
        + " 'PERMX' 'PERMZ'  1 60 1 60 1 7 /\n"
        + "/\nMULTIPLY\n"
        + " 'PERMZ' 0.1  1 60 1 60 1 7 /\n/\n"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        destination.chmod(0o644)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
