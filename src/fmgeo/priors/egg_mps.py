"""Capability-gated multiple-point-statistics generation for Egg."""

from __future__ import annotations

import importlib
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray


class MPSCapabilityError(RuntimeError):
    """Raised when no verified MPS implementation is available."""


MPSBackend = Callable[[NDArray[np.generic], int, int], NDArray[np.generic]]


def _run_mpslib(
    module: Any,
    images: NDArray[np.generic],
    count: int,
    seed: int,
    method: str,
    conditioning_nodes: int,
    executable_dir: Path | None,
) -> NDArray[np.generic]:
    """Run the documented scikit-mps interface in isolated scratch folders."""

    rng = np.random.default_rng(seed)
    source_indices = rng.integers(0, images.shape[0], size=count)
    generated = np.empty((count, *images.shape[1:]), dtype=images.dtype)
    original_directory = os.getcwd()
    try:
        for source_index in np.unique(source_indices):
            output_indices = np.flatnonzero(source_indices == source_index)
            with tempfile.TemporaryDirectory(prefix="fmgeo-mps-") as scratch:
                os.chdir(scratch)
                try:
                    try:
                        simulator = module.mpslib(
                            method=method,
                            simulation_grid_size=np.asarray(images.shape[:0:-1]),
                            n_real=len(output_indices),
                            rseed=seed + int(source_index),
                            n_cond=conditioning_nodes,
                            out_folder=".",
                            verbose_level=-1,
                        )
                        simulator.ti = np.transpose(images[int(source_index)], (2, 1, 0))
                        if executable_dir is not None:
                            simulator.mpslib_exe_folder = str(executable_dir)
                        success = simulator.run(silent=True)
                    except (AttributeError, OSError, RuntimeError) as error:
                        raise MPSCapabilityError(
                            "installed mpslib cannot execute on this platform; use a compatible "
                            "Linux x86_64 environment with scikit-mps and NumPy < 2"
                        ) from error
                finally:
                    os.chdir(original_directory)
                if not success or simulator.sim is None:
                    raise MPSCapabilityError("mpslib did not produce all requested realizations")
                batch = np.stack(
                    [np.transpose(np.asarray(field), (2, 1, 0)) for field in simulator.sim]
                )
                if batch.shape != (len(output_indices), *images.shape[1:]):
                    raise MPSCapabilityError(
                        f"mpslib returned shape {batch.shape}, expected "
                        f"{(len(output_indices), *images.shape[1:])}"
                    )
                generated[output_indices] = batch.astype(images.dtype, copy=False)
    finally:
        os.chdir(original_directory)
    return generated


def generate_mps_realizations(
    training_images: NDArray[np.generic],
    *,
    count: int,
    seed: int,
    backend: MPSBackend | None = None,
    method: Literal["mps_genesim", "mps_snesim_tree"] = "mps_genesim",
    conditioning_nodes: int = 36,
    executable_dir: Path | None = None,
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
    if conditioning_nodes < 1:
        raise ValueError("conditioning_nodes must be positive")

    if backend is None:
        try:
            mpslib = importlib.import_module("mpslib")
        except ImportError:
            try:
                importlib.import_module("geone")
            except ImportError:
                pass
            else:
                raise MPSCapabilityError(
                    "geone is installed but its DeeSse runtime needs a version-tested adapter"
                ) from None
            raise MPSCapabilityError(
                "MPS generation requires geone or mpslib; neither dependency is installed"
            ) from None
        return _run_mpslib(
            mpslib,
            images,
            count,
            seed,
            method,
            conditioning_nodes,
            executable_dir,
        )

    generated = np.asarray(backend(images, count, seed))
    expected_shape = (count, *images.shape[1:])
    if generated.shape != expected_shape:
        raise ValueError(f"MPS backend returned {generated.shape}, expected {expected_shape}")
    return generated
