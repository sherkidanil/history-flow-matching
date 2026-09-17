"""Build the localized 200-member Egg comparison table."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import h5py  # type: ignore[import-untyped]
import numpy as np
from f2_build_remedies import _build_rows, _load_report, _write_csv
from m8_egg_case import load_egg_ensemble

from fmgeo.forward.egg import EGG_PRODUCERS, parse_egg_well_locations
from fmgeo.inverse.egg import load_egg_inversion_config

VARIANTS = (
    "raw_localized",
    "fm_localized",
    "pca_unlocalized",
    "raw_unlocalized",
)


def _parse_labeled_paths(values: list[str]) -> dict[str, Path]:
    mapped: dict[str, Path] = {}
    for value in values:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            raise ValueError("labeled paths must have the form label=path")
        if label in mapped:
            raise ValueError(f"duplicate labeled path: {label}")
        mapped[label] = Path(raw_path)
    return mapped


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--deck", type=Path, required=True)
    parser.add_argument("--realizations-dir", type=Path, required=True)
    parser.add_argument("--active-source", type=Path, required=True)
    parser.add_argument("--evaluation-report", type=Path, required=True)
    parser.add_argument("--inversion", action="append", required=True)
    parser.add_argument("--inversion-report", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--git-commit")
    args = parser.parse_args()

    inversions = _parse_labeled_paths(args.inversion)
    reports = _parse_labeled_paths(args.inversion_report)
    if tuple(inversions) != VARIANTS or tuple(reports) != VARIANTS:
        raise ValueError(f"v2 inputs must be ordered as: {', '.join(VARIANTS)}")
    config = load_egg_inversion_config(args.config)
    if config.ensemble_size != 200:
        raise ValueError("v2 comparison requires an ensemble size of 200")
    evaluation = _load_report(args.evaluation_report)
    threshold = float(evaluation["high_permeability_threshold_logk"])
    with h5py.File(args.active_source) as handle:
        active = np.asarray(handle["active_mask"], dtype=bool)
    truth = load_egg_ensemble(args.realizations_dir)[
        config.truth_realization - 1 : config.truth_realization
    ]
    wells = parse_egg_well_locations(
        args.deck.read_text(encoding="utf-8", errors="replace")
    )
    pair_count = len([name for name in wells if name.startswith("INJECT")]) * len(
        [name for name in wells if name.startswith("PROD")]
    )
    if pair_count != 32 or set(name for name in wells if name.startswith("PROD")) != set(
        EGG_PRODUCERS
    ):
        raise ValueError("official Egg deck must define all 32 injector-producer pairs")
    rows = _build_rows(
        inversions,
        reports=reports,
        truth=truth,
        active=active,
        threshold=threshold,
        wells=wells,
        derived_git_commit=args.git_commit or _git_commit(),
    )
    _write_csv(args.output, rows)
    print(json.dumps({"output": str(args.output), "row_count": len(rows)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
