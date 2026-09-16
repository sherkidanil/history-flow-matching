"""Run a compact, restart-free OPM forward ensemble for Egg fields."""

from __future__ import annotations

import argparse
import json
import subprocess
from functools import partial
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np

from fmgeo.artifacts import (
    canonical_config_hash,
    create_artifact_record,
    sha256_file,
    update_manifest_atomic,
)
from fmgeo.forward.egg import run_egg_forward
from fmgeo.inverse.egg import evaluate_egg_ensemble, load_egg_inversion_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fields", type=Path, required=True)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--flow-command", type=Path, required=True)
    parser.add_argument("--simulator-id", required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--git-commit")
    args = parser.parse_args()
    if args.start_index < 0 or args.count < 1:
        raise ValueError("start index must be non-negative and count must be positive")

    config = load_egg_inversion_config(args.config)
    with h5py.File(args.fields) as handle:
        dataset = handle["logk"]
        stop = args.start_index + args.count
        if stop > len(dataset):
            raise ValueError("requested ensemble slice exceeds the HDF5 dataset")
        fields = np.asarray(dataset[args.start_index:stop], dtype=np.float64)

    evaluate = partial(
        run_egg_forward,
        template_dir=args.template_dir,
        simulator_command=(str(args.flow_command.resolve()),),
        simulator_id=args.simulator_id,
        work_root=args.work_root,
        cache_dir=args.cache_dir,
        history_end_day=config.history_end_day,
        observation_interval_days=config.observation_interval_days,
        oil_rate_relative_sigma=config.oil_rate_relative_sigma,
        water_rate_relative_sigma=config.water_rate_relative_sigma,
        rate_sigma_floor=config.rate_sigma_floor,
        water_breakthrough_fraction=config.water_breakthrough_fraction,
        timeout=config.timeout_seconds,
        min_free_disk_gb=config.min_free_disk_gb,
    )
    result = evaluate_egg_ensemble(fields, evaluator=evaluate, workers=config.workers)
    config_hash = canonical_config_hash(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    string_dtype = h5py.string_dtype(encoding="utf-8")
    with h5py.File(args.output, "w") as handle:
        handle.create_dataset("simulated_data", data=result.simulated_data.astype(np.float32))
        handle.create_dataset("fopt", data=result.fopt.astype(np.float32))
        handle.create_dataset("status", data=result.status.astype(string_dtype))
        handle.create_dataset("runtime_seconds", data=result.runtime_seconds.astype(np.float32))
        handle.create_dataset("cache_hit", data=result.cache_hit)
        handle.create_dataset("returncode", data=result.returncode)
        handle.create_dataset("stderr", data=np.asarray(result.stderr, dtype=string_dtype))
        handle.create_dataset(
            "metadata_json",
            data=np.asarray(
                [json.dumps(item, sort_keys=True) for item in result.metadata],
                dtype=string_dtype,
            ),
        )
        handle.attrs["config_hash"] = config_hash
        handle.attrs["source_sha256"] = sha256_file(args.fields)
        handle.attrs["source_start_index"] = args.start_index

    git_commit = args.git_commit or subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    record = create_artifact_record(
        args.output,
        root=Path.cwd(),
        shape=result.simulated_data.shape,
        dtype="float32",
        config_hash=config_hash,
        git_commit=git_commit,
    )
    update_manifest_atomic(args.manifest, [record])
    successful = result.status == "ok"
    report = {
        "benchmark": "Egg",
        "config_hash": config_hash,
        "source": str(args.fields),
        "source_sha256": sha256_file(args.fields),
        "source_start_index": args.start_index,
        "requested_members": args.count,
        "successful_members": int(successful.sum()),
        "failed_members": int((~successful).sum()),
        "cache_hits": int(result.cache_hit.sum()),
        "runtime_seconds_sum": float(result.runtime_seconds.sum()),
        "simulator_id": args.simulator_id,
        "artifact": record.model_dump(mode="json"),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
