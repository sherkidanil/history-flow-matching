"""Validate the acquired official Egg Eclipse deck."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fmgeo.artifacts import sha256_file
from fmgeo.grids import parse_deck_metadata, validate_benchmark

EXPECTED_WELLS = tuple(
    [f"INJECT{index}" for index in range(1, 9)]
    + [f"PROD{index}" for index in range(1, 5)]
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--actnum", type=Path, required=True)
    args = parser.parse_args()
    metadata = parse_deck_metadata(args.deck, actnum_path=args.actnum)
    validate_benchmark(
        metadata,
        expected_dimensions=(60, 60, 7),
        expected_active_cells=18553,
        expected_wells=EXPECTED_WELLS,
    )
    print(
        json.dumps(
            {
                "benchmark": "Egg",
                "status": "validated",
                "metadata": metadata.as_dict(),
                "files": {
                    str(args.deck): sha256_file(args.deck),
                    str(args.actnum): sha256_file(args.actnum),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
