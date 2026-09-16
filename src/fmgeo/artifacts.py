"""Content-addressed artifacts and provenance-complete manifests."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

SHA256_PATTERN = r"^[0-9a-f]{64}$"
GIT_COMMIT_PATTERN = r"^[0-9a-f]{7,64}$"


class ProvenanceModel(BaseModel):
    """Strict immutable provenance record base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RawProvenance(ProvenanceModel):
    """Execution metadata required for every raw scientific result."""

    git_commit: str = Field(pattern=GIT_COMMIT_PATTERN)
    config_hash: str = Field(pattern=SHA256_PATTERN)
    input_hashes: dict[str, str]
    simulator_version: str = Field(min_length=1)
    platform: dict[str, Any]
    command: tuple[str, ...]
    seed: int = Field(ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("input_hashes")
    @classmethod
    def validate_input_hashes(cls, values: dict[str, str]) -> dict[str, str]:
        invalid = [name for name, value in values.items() if len(value) != 64]
        if invalid:
            raise ValueError(f"input hashes must be SHA-256 values: {invalid}")
        return values


class ArtifactRecord(ProvenanceModel):
    """Manifest metadata read from a materialized artifact."""

    path: str = Field(min_length=1)
    shape: tuple[int, ...]
    dtype: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(pattern=SHA256_PATTERN)
    config_hash: str = Field(pattern=SHA256_PATTERN)
    git_commit: str = Field(pattern=GIT_COMMIT_PATTERN)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("shape")
    @classmethod
    def validate_shape(cls, values: tuple[int, ...]) -> tuple[int, ...]:
        if any(value < 0 for value in values):
            raise ValueError("shape dimensions must be non-negative")
        return values


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash exact file bytes without loading large artifacts into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"unsupported value in canonical configuration: {type(value).__name__}")


def canonical_config_hash(config: Mapping[str, Any] | BaseModel) -> str:
    """Hash a configuration after deterministic JSON normalization."""
    payload: Mapping[str, Any] | dict[str, Any] = (
        config.model_dump(mode="json") if isinstance(config, BaseModel) else config
    )
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def create_artifact_record(
    path: str | Path,
    *,
    root: str | Path,
    shape: Sequence[int],
    dtype: str,
    config_hash: str,
    git_commit: str,
) -> ArtifactRecord:
    """Read an artifact's size and digest into a validated record."""
    artifact_path = Path(path).resolve()
    root_path = Path(root).resolve()
    return ArtifactRecord(
        path=artifact_path.relative_to(root_path).as_posix(),
        shape=tuple(shape),
        dtype=dtype,
        size_bytes=artifact_path.stat().st_size,
        sha256=sha256_file(artifact_path),
        config_hash=config_hash,
        git_commit=git_commit,
    )


def write_manifest_atomic(path: str | Path, records: Sequence[ArtifactRecord]) -> None:
    """Replace a manifest atomically so interrupted writes cannot corrupt it."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "artifacts": [record.model_dump(mode="json") for record in records],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_name = stream.name
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def update_manifest_atomic(path: str | Path, records: Sequence[ArtifactRecord]) -> None:
    """Upsert records by artifact path while retaining existing valid entries."""

    destination = Path(path)
    existing: list[ArtifactRecord] = []
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != 1:
            raise ValueError("existing artifact manifest has an unsupported schema")
        items = payload.get("artifacts")
        if not isinstance(items, list):
            raise ValueError("existing artifact manifest must contain an artifacts list")
        existing = [ArtifactRecord.model_validate(item) for item in items]

    by_path = {record.path: record for record in existing}
    by_path.update({record.path: record for record in records})
    write_manifest_atomic(destination, [by_path[name] for name in sorted(by_path)])
