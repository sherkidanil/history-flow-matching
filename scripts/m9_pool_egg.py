"""Create a provenance-complete horizontally pooled Egg dataset."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np

from fmgeo.artifacts import (
    canonical_config_hash,
    create_artifact_record,
    sha256_file,
    update_manifest_atomic,
)
from fmgeo.resolution import average_pool_horizontal


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--factor", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    if args.factor < 1 or args.batch_size < 1:
        raise ValueError("factor and batch size must be positive")

    source_hash = sha256_file(args.input)
    transform = {
        "schema_version": 1,
        "kind": "active_aware_horizontal_average_pool",
        "factor": args.factor,
        "source_sha256": source_hash,
    }
    config_hash = canonical_config_hash(transform)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.input) as source:
        source_fields = source["logk"]
        active = np.asarray(source["active_mask"], dtype=bool)
        if source_fields.ndim != 4 or source_fields.shape[1:] != active.shape:
            raise ValueError("input logk and active mask shapes do not match")
        input_shape = tuple(source_fields.shape)
        dtype_name = str(source_fields.dtype)
        _, coarse_active = average_pool_horizontal(
            np.zeros((1, *active.shape), dtype=np.float32),
            active_mask=active,
            factor=args.factor,
        )
        output_shape = (len(source_fields), *coarse_active.shape)
        with h5py.File(args.output, "w") as destination:
            fields = destination.create_dataset(
                "logk",
                shape=output_shape,
                dtype=source_fields.dtype,
                chunks=(1, *coarse_active.shape),
                compression="gzip",
                shuffle=True,
            )
            destination.create_dataset("active_mask", data=coarse_active, compression="gzip")
            for start in range(0, len(source_fields), args.batch_size):
                stop = min(start + args.batch_size, len(source_fields))
                pooled, batch_active = average_pool_horizontal(
                    np.asarray(source_fields[start:stop]),
                    active_mask=active,
                    factor=args.factor,
                )
                if not np.array_equal(batch_active, coarse_active):
                    raise RuntimeError("pooled active mask changed between batches")
                fields[start:stop] = pooled
            for name, value in source.attrs.items():
                destination.attrs[name] = value
            destination.attrs["pooling_factor"] = args.factor
            destination.attrs["source_sha256"] = source_hash
            destination.attrs["config_hash"] = config_hash

    git_commit = args.git_commit or subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    record = create_artifact_record(
        args.output,
        root=Path.cwd(),
        shape=output_shape,
        dtype=dtype_name,
        config_hash=config_hash,
        git_commit=git_commit,
    )
    update_manifest_atomic(args.manifest, [record])
    report = {
        "benchmark": "Egg",
        "transform": transform,
        "input": args.input.as_posix(),
        "input_shape": list(input_shape),
        "output_shape": list(output_shape),
        "active_cells_input": int(active.sum()),
        "active_cells_output": int(coarse_active.sum()),
        "artifact": record.model_dump(mode="json"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
