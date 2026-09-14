"""Cross-platform runtime capability detection."""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Literal

import psutil

Device = Literal["auto", "cpu", "mps", "cuda"]


def select_device(
    requested: Device,
    *,
    cuda_available: bool,
    mps_available: bool,
) -> Literal["cpu", "mps", "cuda"]:
    """Resolve an explicit or automatic Torch device selection."""
    if requested == "cuda":
        if not cuda_available:
            raise RuntimeError("CUDA was requested but is unavailable")
        return "cuda"
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("MPS was requested but is unavailable")
        return "mps"
    if requested == "cpu":
        return "cpu"
    if requested != "auto":
        raise ValueError(f"unknown device selection: {requested}")
    if cuda_available:
        return "cuda"
    if mps_available:
        return "mps"
    return "cpu"


def probe_command(command: str) -> dict[str, str | bool | None]:
    """Return command availability and a bounded best-effort version string."""
    path = shutil.which(command)
    if path is None:
        return {"available": False, "path": None, "version": None}
    try:
        completed = subprocess.run(
            [path, "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        output = (completed.stdout or completed.stderr).strip().splitlines()
        version = output[0] if output else None
    except (OSError, subprocess.SubprocessError):
        version = None
    return {"available": True, "path": path, "version": version}


def _probe_torch() -> dict[str, Any]:
    if importlib.util.find_spec("torch") is None:
        return {
            "installed": False,
            "version": None,
            "cuda_available": False,
            "mps_available": False,
            "selected_device": "cpu",
        }

    torch = importlib.import_module("torch")

    cuda_available = bool(torch.cuda.is_available())
    mps_backend = getattr(torch.backends, "mps", None)
    mps_available = bool(mps_backend and mps_backend.is_available())
    return {
        "installed": True,
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "mps_available": mps_available,
        "selected_device": select_device(
            "auto",
            cuda_available=cuda_available,
            mps_available=mps_available,
        ),
    }


def probe_runtime(work_dir: str | Path, *, flow_command: str = "flow") -> dict[str, Any]:
    """Measure portable resources and optional scientific runtimes."""
    resolved_work_dir = Path(work_dir).expanduser().resolve()
    disk_anchor = resolved_work_dir
    while not disk_anchor.exists() and disk_anchor != disk_anchor.parent:
        disk_anchor = disk_anchor.parent
    disk = shutil.disk_usage(disk_anchor)
    scheduler = probe_command("sinfo")
    return {
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "resources": {
            "cpu_count": os.cpu_count() or 1,
            "memory_bytes": psutil.virtual_memory().total,
            "disk_total_bytes": disk.total,
            "disk_free_bytes": disk.free,
            "disk_anchor": str(disk_anchor),
        },
        "torch": _probe_torch(),
        "opm": probe_command(flow_command),
        "scheduler": {
            "type": "slurm" if scheduler["available"] else "none",
            **scheduler,
        },
        "process": {"executable": sys.executable},
    }
