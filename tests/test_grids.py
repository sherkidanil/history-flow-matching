from __future__ import annotations

from pathlib import Path

import pytest

from fmgeo.grids import BenchmarkValidationError, parse_deck_metadata, validate_benchmark


def test_parser_expands_actnum_repeats_and_reads_wells(tmp_path: Path) -> None:
    deck = tmp_path / "MODEL.DATA"
    grid = tmp_path / "GRID.INC"
    deck.write_text(
        """
RUNSPEC
DIMENS
  3 2 1 /
WELSPECS
  'PROD-1' 'G' 1 1 1* 'OIL' /
  'INJ-1'  'G' 3 2 1* 'WATER' /
/
""",
        encoding="utf-8",
    )
    grid.write_text("ACTNUM\n 2*0 3*1 0 /\n", encoding="utf-8")

    metadata = parse_deck_metadata(deck, actnum_path=grid)

    assert metadata.dimensions == (3, 2, 1)
    assert metadata.total_cells == 6
    assert metadata.active_cells == 3
    assert metadata.wells == ("PROD-1", "INJ-1")


def test_grid_dimensions_can_come_from_specgrid_include(tmp_path: Path) -> None:
    deck = tmp_path / "MODEL.DATA"
    grid = tmp_path / "MODEL.GEO"
    deck.write_text("RUNSPEC\nWELSPECS\n 'P1' 'G' 1 1 1* 'OIL' /\n/\n", encoding="utf-8")
    grid.write_text("SPECGRID\n 2 2 1 1 F /\nACTNUM\n 4*1 /\n", encoding="utf-8")

    metadata = parse_deck_metadata(deck, grid_path=grid, actnum_path=grid)

    assert metadata.dimensions == (2, 2, 1)
    assert metadata.active_cells == 4


def test_benchmark_validation_reports_exact_mismatch(tmp_path: Path) -> None:
    deck = tmp_path / "MODEL.DATA"
    deck.write_text(
        "DIMENS\n 2 2 1 /\nACTNUM\n 4*1 /\nWELSPECS\n 'P1' 'G' 1 1 /\n/\n",
        encoding="utf-8",
    )
    metadata = parse_deck_metadata(deck, actnum_path=deck)

    with pytest.raises(BenchmarkValidationError, match="active cells: expected 3, found 4"):
        validate_benchmark(
            metadata,
            expected_dimensions=(2, 2, 1),
            expected_active_cells=3,
            expected_wells=("P1",),
        )

