import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

repository = Path(__file__).parents[1]


def test_g4_controls_builder_exists() -> None:
    assert (repository / "scripts/g4_build_controls.py").is_file()


def _module():
    sys.path.insert(0, str(repository / "scripts"))
    try:
        return importlib.import_module("g4_build_controls")
    finally:
        sys.path.pop(0)


def _write_report(
    path: Path,
    label: str,
    misfits: list[float],
    clipped: list[float] | None,
) -> Path:
    method = label.split("_", maxsplit=1)[0]
    bounded = label not in {"raw_na8", "pca_na8", "fm_na8"}
    localized = "localized" in label
    ensemble_size = 200 if "n200" in label else 100
    stages = []
    for index, misfit in enumerate(misfits):
        row = {"stage": index, "mean_normalized_data_misfit": misfit}
        if clipped is not None:
            row["clipped_fraction"] = clipped[index]
        stages.append(row)
    path.write_text(
        json.dumps(
            {
                "method": method,
                "parameterization_label": label,
                "ensemble_size": ensemble_size,
                "n_sim": ensemble_size * len(stages),
                "localization_enabled": localized,
                "logk_bounds": [2.0, 11.0] if bounded else None,
                "stages": stages,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_stage_rows_preserve_control_order_and_unbounded_clipping_is_missing(
    tmp_path: Path,
) -> None:
    module = _module()
    assert module.G4_VARIANTS == (
        "raw_na8",
        "pca_na8",
        "fm_na8",
        "raw_na8_bounded",
        "pca_na8_bounded",
        "fm_na8_bounded",
        "raw_na8_localized",
        "pca_na8_localized",
        "fm_na8_localized",
        "raw_na8_n200",
        "fm_na8_n200",
    )
    reports = {
        label: _write_report(
            tmp_path / f"{label}.json",
            label,
            [100.0, 10.0],
            None if label in {"raw_na8", "pca_na8", "fm_na8"} else [0.0, 0.25],
        )
        for label in module.G4_VARIANTS
    }

    rows = module._stage_rows(reports)

    assert list(dict.fromkeys(row["variant"] for row in rows)) == list(module.G4_VARIANTS)
    assert [row["clipped_fraction"] for row in rows[:2]] == [None, None]
    bounded_start = 3 * 2
    assert rows[bounded_start + 1]["clipped_fraction"] == 0.25


def test_stage_rows_reject_metadata_that_conflicts_with_variant(tmp_path: Path) -> None:
    module = _module()
    reports = {
        label: _write_report(
            tmp_path / f"{label}.json",
            label,
            [1.0],
            None if label in {"raw_na8", "pca_na8", "fm_na8"} else [0.0],
        )
        for label in module.G4_VARIANTS
    }
    payload = json.loads(reports["pca_na8_localized"].read_text(encoding="utf-8"))
    payload["localization_enabled"] = False
    reports["pca_na8_localized"].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="localization"):
        module._stage_rows(reports)


def test_stage_rows_reject_missing_g4_variant(tmp_path: Path) -> None:
    module = _module()
    reports = {
        label: _write_report(
            tmp_path / f"{label}.json",
            label,
            [1.0],
            None if label in {"raw_na8", "pca_na8", "fm_na8"} else [0.0],
        )
        for label in module.G4_VARIANTS[:-1]
    }

    with pytest.raises(ValueError, match="ordered as"):
        module._stage_rows(reports)


def test_g4_plot_writes_deterministic_svg_and_pdf(tmp_path: Path) -> None:
    module = _module()
    rows = [
        {
            "variant": variant,
            "stage": stage,
            "mean_normalized_data_misfit": 100.0 / (stage + 1),
            "clipped_fraction": (
                None
                if variant in {"raw_na8", "pca_na8", "fm_na8"}
                else 0.01 * stage
            ),
        }
        for variant in module.G4_VARIANTS
        for stage in range(2)
    ]
    svg = tmp_path / "g4.svg"
    pdf = tmp_path / "g4.pdf"

    module._plot(rows, svg, pdf)
    first = hashlib.sha256(svg.read_bytes()).hexdigest()
    module._plot(rows, svg, pdf)

    assert svg.is_file() and pdf.is_file()
    assert hashlib.sha256(svg.read_bytes()).hexdigest() == first
