"""Command-line interface for fmgeo."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level command-line parser without optional imports."""
    return argparse.ArgumentParser(
        prog="fmgeo",
        description="Flow-matching geological inversion",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    parser = build_parser()
    parser.parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

