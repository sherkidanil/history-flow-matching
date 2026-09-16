"""Capability-gated multiple-point-statistics generation for Egg."""

from __future__ import annotations

import importlib
import os
import tempfile
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray


class MPSCapabilityError(RuntimeError):
    """Raised when no verified MPS implementation is available."""


MPSBackend = Callable[[NDArray[np.generic], int, int], NDArray[np.generic]]


@dataclass(frozen=True)
class _MPSJob:
    output_indices: tuple[int, ...]
    image: NDArray[np.generic]
    output_shape: tuple[int, int, int]
    seed: int
    method: str
    conditioning_nodes: int
    executable_dir: Path | None
    template_size_xyz: tuple[int, int, int]
    multiple_grids: int


def _run_mpslib_job(job: _MPSJob) -> tuple[tuple[int, ...], NDArray[np.generic]]:
    """Run one training image in an isolated worker and scratch folder."""

    module = importlib.import_module("mpslib")
    original_directory = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="fmgeo-mps-") as scratch:
        os.chdir(scratch)
        try:
            try:
                simulator = module.mpslib(
                    method=job.method,
                    simulation_grid_size=np.asarray(job.output_shape[::-1]),
                    n_real=len(job.output_indices),
                    rseed=job.seed,
                    n_cond=job.conditioning_nodes,
                    template_size=np.asarray(job.template_size_xyz),
                    n_multiple_grids=job.multiple_grids,
                    out_folder=".",
                    verbose_level=-1,
                )
                simulator.ti = np.transpose(job.image, (2, 1, 0))
                if job.executable_dir is not None:
                    simulator.mpslib_exe_folder = str(job.executable_dir)
                success = simulator.run(silent=True)
            except Exception as error:
                raise MPSCapabilityError(
                    "installed mpslib cannot execute on this platform; use a compatible "
                    "Linux x86_64 environment with scikit-mps and NumPy < 2"
                ) from error
        finally:
            os.chdir(original_directory)
    if not success or simulator.sim is None:
        raise MPSCapabilityError("mpslib did not produce all requested realizations")
    batch = np.stack([np.transpose(np.asarray(field), (2, 1, 0)) for field in simulator.sim])
    expected_shape = (len(job.output_indices), *job.output_shape)
    if batch.shape != expected_shape:
        raise MPSCapabilityError(f"mpslib returned shape {batch.shape}, expected {expected_shape}")
    return job.output_indices, batch.astype(job.image.dtype, copy=False)


def _run_mpslib(
    images: NDArray[np.generic],
    count: int,
    seed: int,
    method: str,
    conditioning_nodes: int,
    executable_dir: Path | None,
    template_size_xyz: tuple[int, int, int],
    multiple_grids: int,
    workers: int,
) -> NDArray[np.generic]:
    """Run independent training images in parallel isolated processes."""

    rng = np.random.default_rng(seed)
    source_indices = rng.integers(0, images.shape[0], size=count)
    jobs = [
        _MPSJob(
            output_indices=tuple(int(index) for index in np.flatnonzero(source_indices == source)),
            image=images[int(source)],
            output_shape=images.shape[1:],
            seed=seed + int(source),
            method=method,
            conditioning_nodes=conditioning_nodes,
            executable_dir=executable_dir,
            template_size_xyz=template_size_xyz,
            multiple_grids=multiple_grids,
        )
        for source in np.unique(source_indices)
    ]
    generated = np.empty((count, *images.shape[1:]), dtype=images.dtype)
    if workers == 1:
        for output_indices, batch in map(_run_mpslib_job, jobs):
            generated[list(output_indices)] = batch
        return generated

    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as executor:
        for output_indices, batch in executor.map(_run_mpslib_job, jobs):
            generated[list(output_indices)] = batch
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
    template_size_xyz: tuple[int, int, int] = (12, 12, 3),
    multiple_grids: int = 2,
    workers: int = 1,
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
    if any(size < 1 for size in template_size_xyz) or multiple_grids < 0:
        raise ValueError("MPS template sizes must be positive and multiple_grids non-negative")
    if workers < 1:
        raise ValueError("workers must be positive")

    if backend is None:
        try:
            importlib.import_module("mpslib")
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
        if executable_dir is not None:
            executable_dir = executable_dir.expanduser().resolve()
            executable = executable_dir / method
            if not executable.is_file() or not os.access(executable, os.X_OK):
                raise MPSCapabilityError(
                    f"validated MPSlib executable is missing or not executable: {executable}"
                )
        return _run_mpslib(
            images,
            count,
            seed,
            method,
            conditioning_nodes,
            executable_dir,
            template_size_xyz,
            multiple_grids,
            workers,
        )

    generated = np.asarray(backend(images, count, seed))
    expected_shape = (count, *images.shape[1:])
    if generated.shape != expected_shape:
        raise ValueError(f"MPS backend returned {generated.shape}, expected {expected_shape}")
    return generated
