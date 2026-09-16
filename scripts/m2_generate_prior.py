"""Generate and validate a conditioned PUNQ-S3 object-based prior artifact."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np

from fmgeo.artifacts import (
    canonical_config_hash,
    create_artifact_record,
    sha256_file,
    write_manifest_atomic,
)
from fmgeo.grids import parse_numeric_keyword
from fmgeo.priors.punq_objectbased import (
    load_punq_prior_config,
    validate_prior_against_truth,
    write_punq_prior_hdf5,
)


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _accepted_coverage_report(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("coverage report must be a JSON object")
    if (
        payload.get("accepted") is not True
        or int(payload.get("forward_runs", 0)) < 100
        or payload.get("all_six_wells_covered") is not True
    ):
        raise ValueError(
            "full prior requires an accepted 100-run, six-well forward coverage report"
        )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--truth-properties", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--coverage-report", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_punq_prior_config(args.config)
    if args.count > 1000:
        if args.coverage_report is None:
            raise ValueError("--coverage-report is required before full prior generation")
        _accepted_coverage_report(args.coverage_report)

    active_values = parse_numeric_keyword(args.grid, "ACTNUM")
    if len(active_values) != int(np.prod(config.shape)):
        raise ValueError("ACTNUM length does not match configured PUNQ grid")
    active = np.asarray(active_values, dtype=bool).reshape(config.shape)
    if config.shape == (5, 28, 19) and int(active.sum()) != 1761:
        raise ValueError("official PUNQ-S3 grid must contain 1761 active cells")

    metadata = write_punq_prior_hdf5(
        args.output,
        config=config,
        active_mask=active,
        count=args.count,
        seed=config.seed,
    )
    truth_poro = np.asarray(
        parse_numeric_keyword(args.truth_properties, "PORO"), dtype=np.float64
    ).reshape(config.shape)
    truth_logk = np.log(
        np.asarray(
            parse_numeric_keyword(args.truth_properties, "PERMX"), dtype=np.float64
        ).reshape(config.shape)
    )
    with h5py.File(args.output) as handle:
        validation = validate_prior_against_truth(
            config=config,
            active_mask=active,
            facies_ensemble=np.asarray(handle["facies"]),
            logk_ensemble=np.asarray(handle["logk"]),
            truth_logk=truth_logk,
            truth_porosity=truth_poro,
        )

    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    config_hash = canonical_config_hash(config)
    artifact_record = create_artifact_record(
        args.output,
        root=Path.cwd(),
        shape=(args.count, *config.shape),
        dtype=str(metadata["dtype"]),
        config_hash=config_hash,
        git_commit=git_commit,
    )
    write_manifest_atomic(args.manifest, [artifact_record])
    report: dict[str, object] = {
        "benchmark": "PUNQ-S3",
        "count": args.count,
        "seed": config.seed,
        "config_hash": config_hash,
        "input_hashes": {
            str(args.grid): sha256_file(args.grid),
            str(args.truth_properties): sha256_file(args.truth_properties),
        },
        "artifact": artifact_record.model_dump(mode="json"),
        "validation": validation,
    }
    _write_json_atomic(args.validation_report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if validation["accepted"] is True else 3


if __name__ == "__main__":
    raise SystemExit(main())
