"""Deterministic vector-figure output shared by publication scripts."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure


def save_vector_figure(
    figure: Figure,
    path: str | Path,
    *,
    creator: str,
    description: str | None = None,
) -> None:
    """Save SVG or PDF without wall-clock metadata."""
    destination = Path(path)
    vector_format = destination.suffix.removeprefix(".").lower()
    if vector_format == "svg":
        mpl.rcParams["svg.hashsalt"] = f"fmgeo:{creator}"
        metadata: dict[str, object] = {"Creator": creator, "Date": None}
        if description is not None:
            metadata["Description"] = description
    elif vector_format == "pdf":
        metadata = {"Creator": creator, "CreationDate": None, "ModDate": None}
        if description is not None:
            metadata["Subject"] = description
    else:
        raise ValueError("publication figure output must end in .svg or .pdf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, format=vector_format, metadata=metadata)
