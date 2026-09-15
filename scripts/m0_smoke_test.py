"""Probe a host and run the storage-safe OPM Flow SPE1 acceptance test."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any

from fmgeo.forward.deck import ensure_summary_keywords, strip_restart_output
from fmgeo.forward.observables import read_summary_vectors
from fmgeo.forward.runner import require_free_disk
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
    lines = [
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
    ]
    smoke = result.get("smoke")
    if isinstance(smoke, dict):
        lines.extend(
            [
                f"- SPE1 smoke status: {smoke.get('status')}",
                f"- SPE1 runtime seconds: {smoke.get('runtime_seconds')}",
                f"- SPE1 final FOPT: {smoke.get('fopt_final')}",
                f"- SPE1 SMSPEC bytes: {smoke.get('smspec_bytes')}",
                f"- SPE1 UNSMRY bytes: {smoke.get('unsmry_bytes')}",
                f"- SPE1 total output bytes: {smoke.get('total_output_bytes')}",
                "- Transient simulator disk free bytes before run: "
                f"{smoke.get('transient_disk_free_bytes_before')}",
                f"- SPE1 restart output present: {smoke.get('restart_output_present')}",
            ]
        )
    lines.extend([END_MARKER, ""])
    section = "\n".join(lines)
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
    parser.add_argument("--spe1-deck", type=Path)
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument("--minimum-free-disk-gb", type=float, default=20.0)
    parser.add_argument(
        "--disk-check-path",
        type=Path,
        help="filesystem holding transient simulator output (for example Docker storage)",
    )
    parser.add_argument(
        "--environment-file",
        type=Path,
        help="replace the generated probe section in this Markdown file",
    )
    return parser


def run_spe1_smoke(
    *,
    flow_command: str,
    source_deck: Path,
    work_root: Path,
    disk_check_path: Path | None,
    minimum_free_disk_gb: float,
    keep_output: bool,
) -> dict[str, Any]:
    """Run a sanitized SPE1 deck and validate its compact summary output."""
    transient_disk = disk_check_path or work_root
    require_free_disk(transient_disk, minimum_free_disk_gb)
    transient_free_before = shutil.disk_usage(transient_disk).free
    work_root.mkdir(parents=True, exist_ok=True)
    run_directory = Path(tempfile.mkdtemp(prefix="spe1-", dir=work_root)).resolve()
    try:
        rendered = strip_restart_output(source_deck.read_text(encoding="utf-8"))
        rendered = ensure_summary_keywords(rendered, ["FOPT"])
        deck_path = run_directory / "SPE1CASE1.DATA"
        deck_path.write_text(rendered, encoding="utf-8")
        started = time.perf_counter()
        completed = subprocess.run(
            [flow_command, deck_path.name],
            cwd=run_directory,
            check=False,
            capture_output=True,
            text=True,
            timeout=600,
        )
        runtime = time.perf_counter() - started
        if completed.returncode != 0:
            detail = completed.stderr.strip().splitlines()[-1:] or ["no stderr"]
            raise RuntimeError(
                f"Flow exited with code {completed.returncode}: {detail[0]}"
            )
        smspec = run_directory / "SPE1CASE1.SMSPEC"
        unsmry = run_directory / "SPE1CASE1.UNSMRY"
        if not smspec.is_file() or not unsmry.is_file():
            raise RuntimeError("Flow did not produce both SMSPEC and UNSMRY")
        restart_files = list(run_directory.glob("*.UNRST")) + list(
            run_directory.glob("*.X[0-9][0-9][0-9][0-9]")
        )
        if restart_files:
            raise RuntimeError("sanitized SPE1 run unexpectedly produced restart output")
        summary = read_summary_vectors(run_directory / "SPE1CASE1", ["FOPT"])
        return {
            "status": "ok",
            "runtime_seconds": runtime,
            "fopt_final": float(summary["FOPT"][-1]),
            "smspec_bytes": smspec.stat().st_size,
            "unsmry_bytes": unsmry.stat().st_size,
            "total_output_bytes": smspec.stat().st_size + unsmry.stat().st_size,
            "transient_disk_free_bytes_before": transient_free_before,
            "restart_output_present": False,
            "output_directory": str(run_directory) if keep_output else None,
        }
    finally:
        if not keep_output:
            shutil.rmtree(run_directory)


def main() -> int:
    args = build_parser().parse_args()
    result = probe_runtime(args.work_dir, flow_command=args.flow_command)
    status = 0
    if not args.probe_only:
        if not result["opm"]["available"]:
            result["smoke"] = {"status": "failed", "error": "OPM Flow unavailable"}
            status = 2
        elif args.spe1_deck is None:
            result["smoke"] = {"status": "failed", "error": "--spe1-deck is required"}
            status = 2
        else:
            try:
                result["smoke"] = run_spe1_smoke(
                    flow_command=args.flow_command,
                    source_deck=args.spe1_deck,
                    work_root=args.work_dir,
                    disk_check_path=args.disk_check_path,
                    minimum_free_disk_gb=args.minimum_free_disk_gb,
                    keep_output=args.keep_output,
                )
            except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
                result["smoke"] = {"status": "failed", "error": str(error)}
                status = 1
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.environment_file is not None:
        write_environment_snapshot(args.environment_file, result)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
