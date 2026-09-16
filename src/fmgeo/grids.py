"""Minimal, auditable metadata parsing for Eclipse benchmark decks."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


class BenchmarkValidationError(ValueError):
    """Raised when measured deck metadata violates an acceptance criterion."""


@dataclass(frozen=True)
class DeckMetadata:
    """Grid and well facts extracted without executing a simulator."""

    dimensions: tuple[int, int, int]
    active_cells: int
    wells: tuple[str, ...]

    @property
    def total_cells(self) -> int:
        nx, ny, nz = self.dimensions
        return nx * ny * nz

    def as_dict(self) -> dict[str, object]:
        return {**asdict(self), "total_cells": self.total_cells}


def _without_comments(text: str) -> str:
    return "\n".join(line.split("--", maxsplit=1)[0] for line in text.splitlines())


def _keyword_body(text: str, keyword: str) -> str:
    cleaned = _without_comments(text)
    match = re.search(
        rf"(?ims)^\s*{re.escape(keyword)}\s*$\s*(.*?)/",
        cleaned,
    )
    if match is None:
        raise BenchmarkValidationError(f"keyword {keyword} was not found")
    return match.group(1)


def _numeric_values(body: str) -> list[float]:
    values: list[float] = []
    for token in body.replace(",", " ").split():
        if "*" in token:
            count_text, value_text = token.split("*", maxsplit=1)
            if not count_text.isdigit() or not value_text:
                raise BenchmarkValidationError(f"unsupported defaulted repeat token: {token}")
            values.extend([float(value_text)] * int(count_text))
        else:
            values.append(float(token))
    return values


def parse_numeric_keyword(path: str | Path, keyword: str) -> tuple[float, ...]:
    """Read and expand one numeric Eclipse keyword record from a text include."""
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    return tuple(_numeric_values(_keyword_body(text, keyword)))


def _dimensions(deck_text: str, grid_text: str | None) -> tuple[int, int, int]:
    for text, keyword in ((deck_text, "DIMENS"), (grid_text, "SPECGRID")):
        if text is None:
            continue
        try:
            body = _keyword_body(text, keyword)
        except BenchmarkValidationError:
            continue
        dimension_tokens = body.replace(",", " ").split()[:3]
        values = _numeric_values(" ".join(dimension_tokens))
        if len(values) < 3:
            raise BenchmarkValidationError(f"{keyword} contains fewer than three dimensions")
        return tuple(int(value) for value in values[:3])  # type: ignore[return-value]
    raise BenchmarkValidationError("neither DIMENS nor SPECGRID was found")


def _well_names(deck_text: str) -> tuple[str, ...]:
    cleaned = _without_comments(deck_text)
    start = re.search(r"(?im)^\s*WELSPECS\s*$", cleaned)
    if start is None:
        raise BenchmarkValidationError("keyword WELSPECS was not found")
    names: list[str] = []
    for line in cleaned[start.end() :].splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped == "/":
            break
        match = re.match(r"\s*(?:'([^']+)'|([^\s/]+))", line)
        if match is not None:
            names.append(match.group(1) or match.group(2))
    if not names:
        raise BenchmarkValidationError("WELSPECS contains no wells")
    return tuple(names)


def parse_deck_metadata(
    deck_path: str | Path,
    *,
    grid_path: str | Path | None = None,
    actnum_path: str | Path | None = None,
) -> DeckMetadata:
    """Extract dimensions, ACTNUM count, and WELSPECS names from text files."""
    deck = Path(deck_path).read_text(encoding="utf-8", errors="replace")
    grid = (
        Path(grid_path).read_text(encoding="utf-8", errors="replace")
        if grid_path is not None
        else None
    )
    active_source = (
        Path(actnum_path).read_text(encoding="utf-8", errors="replace")
        if actnum_path
        else (grid or deck)
    )
    dimensions = _dimensions(deck, grid)
    actnum = _numeric_values(_keyword_body(active_source, "ACTNUM"))
    expected_cells = dimensions[0] * dimensions[1] * dimensions[2]
    if len(actnum) != expected_cells:
        raise BenchmarkValidationError(
            f"ACTNUM length: expected {expected_cells}, found {len(actnum)}"
        )
    if any(value not in (0.0, 1.0) for value in actnum):
        raise BenchmarkValidationError("ACTNUM must contain only zero and one")
    return DeckMetadata(
        dimensions=dimensions,
        active_cells=int(sum(actnum)),
        wells=_well_names(deck),
    )


def validate_benchmark(
    metadata: DeckMetadata,
    *,
    expected_dimensions: tuple[int, int, int],
    expected_active_cells: int,
    expected_wells: tuple[str, ...],
) -> None:
    """Raise one evidence-rich error for the first failed acceptance criterion."""
    if metadata.dimensions != expected_dimensions:
        raise BenchmarkValidationError(
            f"dimensions: expected {expected_dimensions}, found {metadata.dimensions}"
        )
    if metadata.active_cells != expected_active_cells:
        raise BenchmarkValidationError(
            f"active cells: expected {expected_active_cells}, found {metadata.active_cells}"
        )
    if set(metadata.wells) != set(expected_wells):
        raise BenchmarkValidationError(
            f"wells: expected {sorted(expected_wells)}, found {sorted(metadata.wells)}"
        )
