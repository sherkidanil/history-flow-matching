"""Prepare one restart-free Egg deck for a storage-safe OPM Flow run."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np

from fmgeo.artifacts import sha256_file
from fmgeo.forward.deck import assert_restart_disabled, strip_restart_output
from fmgeo.priors.egg_io import write_egg_permeability_include


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eclipse-dir", type=Path, required=True)
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--permeability",
        type=Path,
        help="optional PERMX realization; defaults to the standard mDARCY include",
    )
    source.add_argument("--sample-hdf5", type=Path, help="FM or prior log-k artifact")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_deck = args.eclipse_dir / "Egg_Model_ECL.DATA"
    if args.sample_index < 0:
        raise ValueError("sample index must be non-negative")
    permeability = args.permeability or args.eclipse_dir / "mDARCY.INC"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    deck = strip_restart_output(source_deck.read_text(encoding="utf-8", errors="replace"))
    assert_restart_disabled(deck)
    output_deck = args.output_dir / "EGG.DATA"
    output_deck.write_text(deck, encoding="utf-8")
    copied = {
        "ACTIVE.INC": args.eclipse_dir / "ACTIVE.INC",
        "COMPDAT.INC": args.eclipse_dir / "COMPDAT.INC",
        "SCHEDULE_NEW.INC": args.eclipse_dir / "SCHEDULE_NEW.INC",
    }
    for name, source in copied.items():
        shutil.copy2(source, args.output_dir / name)
    if args.sample_hdf5 is None:
        shutil.copy2(permeability, args.output_dir / "mDARCY.INC")
    else:
        with h5py.File(args.sample_hdf5) as handle:
            dataset = handle["logk"]
            if args.sample_index >= len(dataset):
                raise ValueError("sample index is outside the HDF5 logk dataset")
            logk = np.asarray(dataset[args.sample_index], dtype=np.float64)
        write_egg_permeability_include(logk, args.output_dir / "mDARCY.INC")

    report = {
        "benchmark": "Egg",
        "restart_output": False,
        "deck": str(output_deck),
        "input_hashes": {
            str(source_deck): sha256_file(source_deck),
            **{str(source): sha256_file(source) for source in copied.values()},
            str(args.sample_hdf5 or permeability): sha256_file(
                args.sample_hdf5 or permeability
            ),
        },
        "output_hashes": {
            str(path): sha256_file(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file()
        },
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
