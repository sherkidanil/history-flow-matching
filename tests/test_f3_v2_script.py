from __future__ import annotations

import sys
from pathlib import Path

import pytest

repository = Path(__file__).parents[1]
sys.path.insert(0, str(repository / "scripts"))
from f3_build_inversion_v2 import _parse_labeled_paths  # noqa: E402

sys.path.pop(0)


def test_parse_labeled_paths_preserves_four_unique_variants() -> None:
    mapped = _parse_labeled_paths(
        [
            "raw_localized=artifacts/raw_loc.h5",
            "fm_localized=artifacts/fm_loc.h5",
            "pca_unlocalized=artifacts/pca.h5",
            "raw_unlocalized=artifacts/raw.h5",
        ]
    )

    assert list(mapped) == [
        "raw_localized",
        "fm_localized",
        "pca_unlocalized",
        "raw_unlocalized",
    ]
    assert mapped["raw_localized"] == Path("artifacts/raw_loc.h5")


def test_parse_labeled_paths_rejects_duplicates() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        _parse_labeled_paths(["raw=a.h5", "raw=b.h5"])
