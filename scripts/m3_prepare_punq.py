"""Render an OPM-compatible, restart-free PUNQ-S3 deck template."""

from __future__ import annotations

import argparse
from pathlib import Path

from fmgeo.forward.deck import render_punq_opm_compatibility, strip_restart_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rendered = render_punq_opm_compatibility(
        strip_restart_output(args.source.read_text(encoding="utf-8"))
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
