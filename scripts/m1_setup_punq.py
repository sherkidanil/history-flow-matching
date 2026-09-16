"""Validate the acquired official PUNQ-S3 text deck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fmgeo.artifacts import sha256_file
from fmgeo.grids import parse_deck_metadata, validate_benchmark

EXPECTED_WELLS = ("PRO-1", "PRO-4", "PRO-5", "PRO-11", "PRO-12", "PRO-15")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    args = parser.parse_args()
    metadata = parse_deck_metadata(args.deck, grid_path=args.grid, actnum_path=args.grid)
    validate_benchmark(
        metadata,
        expected_dimensions=(19, 28, 5),
        expected_active_cells=1761,
        expected_wells=EXPECTED_WELLS,
    )
    print(
        json.dumps(
            {
                "benchmark": "PUNQ-S3",
                "status": "validated",
                "metadata": metadata.as_dict(),
                "files": {
                    str(args.deck): sha256_file(args.deck),
                    str(args.grid): sha256_file(args.grid),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

