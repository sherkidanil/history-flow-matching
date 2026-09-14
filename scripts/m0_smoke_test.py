"""Inspect the current host before running the full M0 simulator smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fmgeo.runtime import probe_runtime

BEGIN_MARKER = "<!-- BEGIN AUTO-GENERATED M0 PROBE -->"
END_MARKER = "<!-- END AUTO-GENERATED M0 PROBE -->"


def write_environment_snapshot(path: Path, result: dict[str, object]) -> None:
    """Replace the generated M0 section while preserving handwritten notes."""
    platform = result["platform"]
    resources = result["resources"]
    opm = result["opm"]
    torch = result["torch"]
    assert isinstance(platform, dict)
    assert isinstance(resources, dict)
    assert isinstance(opm, dict)
    assert isinstance(torch, dict)
    section = "\n".join(
        [
            BEGIN_MARKER,
            "## Latest measured M0 probe",
            "",
            f"- Operating system: {platform['system']} {platform['release']}",
            f"- Machine architecture: {platform['machine']}",
            f"- Python: {platform['python']}",
            f"- CPU count: {resources['cpu_count']}",
            f"- Memory bytes: {resources['memory_bytes']}",
            f"- Free disk bytes: {resources['disk_free_bytes']}",
            f"- OPM Flow available: {opm['available']}",
            f"- OPM Flow version: {opm['version']}",
            f"- Torch installed: {torch['installed']}",
            f"- Selected Torch device: {torch['selected_device']}",
            END_MARKER,
            "",
        ]
    )
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if BEGIN_MARKER in existing and END_MARKER in existing:
        prefix, remainder = existing.split(BEGIN_MARKER, maxsplit=1)
        _, suffix = remainder.split(END_MARKER, maxsplit=1)
        updated = prefix.rstrip() + "\n\n" + section + suffix.lstrip("\n")
    else:
        updated = existing.rstrip() + ("\n\n" if existing.strip() else "") + section
    path.write_text(updated, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="report capabilities without requiring OPM Flow",
    )
    parser.add_argument("--flow-command", default="flow")
    parser.add_argument("--work-dir", type=Path, default=Path("scratch/m0"))
    parser.add_argument(
        "--environment-file",
        type=Path,
        help="replace the generated probe section in this Markdown file",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = probe_runtime(args.work_dir, flow_command=args.flow_command)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.environment_file is not None:
        write_environment_snapshot(args.environment_file, result)
    if not args.probe_only and not result["opm"]["available"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
