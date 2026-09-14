from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fmgeo.artifacts import (
    ArtifactRecord,
    RawProvenance,
    canonical_config_hash,
    create_artifact_record,
    sha256_file,
    write_manifest_atomic,
)


def test_file_hash_depends_on_exact_content(tmp_path: Path) -> None:
    left = tmp_path / "left.bin"
    right = tmp_path / "right.bin"
    left.write_bytes(b"measured-result-a")
    right.write_bytes(b"measured-result-b")

    assert sha256_file(left) != sha256_file(right)
    assert len(sha256_file(left)) == 64


def test_config_hash_is_canonical_across_mapping_order() -> None:
    first = {"seed": 7, "nested": {"alpha": [4.0, 4.0], "path": Path("data")}}
    second = {"nested": {"path": Path("data"), "alpha": [4.0, 4.0]}, "seed": 7}

    assert canonical_config_hash(first) == canonical_config_hash(second)


def test_artifact_record_requires_complete_provenance() -> None:
    with pytest.raises(ValidationError):
        ArtifactRecord.model_validate({"path": "sample.h5", "shape": [2, 3]})

    with pytest.raises(ValidationError):
        RawProvenance.model_validate(
            {
                "git_commit": "abc123",
                "config_hash": "0" * 64,
                "input_hashes": {},
            }
        )


def test_create_artifact_record_reads_file_metadata(tmp_path: Path) -> None:
    artifact = tmp_path / "ensemble.bin"
    artifact.write_bytes(b"real bytes")

    record = create_artifact_record(
        artifact,
        root=tmp_path,
        shape=(1, 10),
        dtype="float32",
        config_hash="a" * 64,
        git_commit="0123456789abcdef",
    )

    assert record.path == "ensemble.bin"
    assert record.size_bytes == len(b"real bytes")
    assert record.sha256 == sha256_file(artifact)


def test_manifest_write_is_atomic_and_valid_json(tmp_path: Path) -> None:
    destination = tmp_path / "MANIFEST.json"
    destination.write_text("old content", encoding="utf-8")
    record = ArtifactRecord(
        path="ensemble.h5",
        shape=(10, 5, 28, 19),
        dtype="float32",
        size_bytes=100,
        sha256="b" * 64,
        config_hash="c" * 64,
        git_commit="0123456789abcdef",
    )

    write_manifest_atomic(destination, [record])

    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["artifacts"][0]["path"] == "ensemble.h5"
    assert list(tmp_path.glob(".MANIFEST.json.*.tmp")) == []

