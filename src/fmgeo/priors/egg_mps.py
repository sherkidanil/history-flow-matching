"""Capability-gated multiple-point-statistics generation for Egg."""

from __future__ import annotations

import importlib
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray


class MPSCapabilityError(RuntimeError):
    """Raised when no verified MPS implementation is available."""


MPSBackend = Callable[[NDArray[np.generic], int, int], NDArray[np.generic]]


def generate_mps_realizations(
    training_images: NDArray[np.generic],
    *,
    count: int,
    seed: int,
    backend: MPSBackend | None = None,
) -> NDArray[np.generic]:
    """Generate MPS realizations using an explicitly verified adapter.

    The Python APIs exposed by geone and mpslib differ substantially between
    releases.  We therefore refuse to silently replace MPS with another prior:
    callers must supply a tested adapter for the installed backend.
    """

    images = np.asarray(training_images)
    if images.ndim != 4:
        raise ValueError("training_images must have shape (image, z, y, x)")
    if count < 1:
        raise ValueError("count must be positive")

    if backend is None:
        installed: list[str] = []
        for package in ("geone", "mpslib"):
            try:
                importlib.import_module(package)
            except ImportError:
                continue
            installed.append(package)
        if not installed:
            raise MPSCapabilityError(
                "MPS generation requires geone or mpslib; neither dependency is installed"
            )
        raise MPSCapabilityError(
            "MPS dependency found ("
            + ", ".join(installed)
            + "), but a version-tested backend adapter is required"
        )

    generated = np.asarray(backend(images, count, seed))
    expected_shape = (count, *images.shape[1:])
    if generated.shape != expected_shape:
        raise ValueError(f"MPS backend returned {generated.shape}, expected {expected_shape}")
    return generated
