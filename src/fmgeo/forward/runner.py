"""Isolated, cache-aware, storage-safe external simulator execution."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, cast


class DiskSpaceError(RuntimeError):
    """Raised before launch when the configured disk reserve is unavailable."""


@dataclass(frozen=True)
class ForwardResult:
    """Measured result of one simulator invocation."""

    status: Literal["ok", "failed", "timeout"]
    d: tuple[float, ...] | None
    fopt_16_5y: float | None
    runtime_seconds: float
    returncode: int | None
    stderr: str
    cache_hit: bool = False
    workdir: str | None = None
    metadata: dict[str, object] | None = None


Extractor = Callable[[Path], dict[str, object]]
Preparer = Callable[[Path], None]


def _disk_anchor(path: Path) -> Path:
    anchor = path.resolve()
    while not anchor.exists() and anchor != anchor.parent:
        anchor = anchor.parent
    return anchor


def require_free_disk(path: str | Path, minimum_gb: float) -> None:
    """Fail before launch if a filesystem has less than the reserved space."""
    if minimum_gb < 0:
        raise ValueError("minimum_gb must be non-negative")
    free_bytes = shutil.disk_usage(_disk_anchor(Path(path))).free
    required_bytes = minimum_gb * 1024**3
    if free_bytes < required_bytes:
        raise DiskSpaceError(
            f"free disk {free_bytes / 1024**3:.2f} GiB is below required "
            f"{minimum_gb:.2f} GiB"
        )


def _load_cache(path: Path) -> ForwardResult | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "ok":
        return None
    payload["d"] = tuple(float(value) for value in payload["d"])
    payload["cache_hit"] = True
    return ForwardResult(**payload)


def _write_cache(path: Path, result: ForwardResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    payload["workdir"] = None
    payload["cache_hit"] = False
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = stream.name
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def run_simulator(
    command: Sequence[str],
    *,
    work_root: str | Path,
    cache_dir: str | Path,
    cache_key: str,
    extractor: Extractor,
    prepare: Preparer | None = None,
    timeout: float = 600.0,
    min_free_disk_gb: float = 20.0,
    disk_check_path: str | Path | None = None,
    keep_workdir: bool = False,
) -> ForwardResult:
    """Execute one simulator in an isolated directory and cache observations only."""
    if not re.fullmatch(r"[0-9a-f]{64}", cache_key):
        raise ValueError("cache_key must be a lowercase SHA-256 digest")
    if timeout <= 0:
        raise ValueError("timeout must be positive")
    require_free_disk(disk_check_path or work_root, min_free_disk_gb)
    cache_path = Path(cache_dir) / f"{cache_key}.json"
    cached = _load_cache(cache_path)
    if cached is not None:
        return cached

    root = Path(work_root)
    root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="fmgeo-", dir=root))
    started = time.perf_counter()
    result: ForwardResult
    try:
        if prepare is not None:
            prepare(workdir)
        completed = subprocess.run(
            list(command),
            cwd=workdir,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        runtime = time.perf_counter() - started
        if completed.returncode != 0:
            result = ForwardResult(
                status="failed",
                d=None,
                fopt_16_5y=None,
                runtime_seconds=runtime,
                returncode=completed.returncode,
                stderr=completed.stderr,
            )
        else:
            try:
                extracted = extractor(workdir)
                raw_observations = cast(Sequence[float], extracted["d"])
                observations = tuple(float(value) for value in raw_observations)
                fopt = float(cast(float, extracted["FOPT_16.5y"]))
                result = ForwardResult(
                    status="ok",
                    d=observations,
                    fopt_16_5y=fopt,
                    runtime_seconds=runtime,
                    returncode=completed.returncode,
                    stderr=completed.stderr,
                    metadata=cast(dict[str, object] | None, extracted.get("metadata")),
                )
                _write_cache(cache_path, result)
            except (KeyError, TypeError, ValueError, OSError) as error:
                result = ForwardResult(
                    status="failed",
                    d=None,
                    fopt_16_5y=None,
                    runtime_seconds=runtime,
                    returncode=completed.returncode,
                    stderr=f"observation extraction failed: {error}",
                )
    except subprocess.TimeoutExpired as error:
        stderr = error.stderr.decode() if isinstance(error.stderr, bytes) else (error.stderr or "")
        result = ForwardResult(
            status="timeout",
            d=None,
            fopt_16_5y=None,
            runtime_seconds=time.perf_counter() - started,
            returncode=None,
            stderr=stderr,
        )
    except (OSError, ValueError) as error:
        result = ForwardResult(
            status="failed",
            d=None,
            fopt_16_5y=None,
            runtime_seconds=time.perf_counter() - started,
            returncode=None,
            stderr=str(error),
        )
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir)
    return replace(result, workdir=str(workdir) if keep_workdir else None)
