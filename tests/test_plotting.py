from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from fmgeo.plotting import save_vector_figure


def test_vector_figure_writer_is_deterministic_for_svg_and_pdf(tmp_path: Path) -> None:
    figure, axis = plt.subplots()
    axis.plot([0.0, 1.0], [1.0, 0.0])
    for suffix, magic in (("svg", b"<?xml"), ("pdf", b"%PDF")):
        output = tmp_path / f"figure.{suffix}"
        save_vector_figure(figure, output, creator="test")
        first = output.read_bytes()
        save_vector_figure(figure, output, creator="test")
        assert output.read_bytes() == first
        assert first.startswith(magic)
    plt.close(figure)
