from __future__ import annotations

import numpy as np
import pytest

from fmgeo.inverse.localization import project_localization_to_pca


def test_project_localization_to_pca_uses_squared_loading_weights() -> None:
    cell_taper = np.array(
        [
            [1.0, 0.2],
            [0.5, 1.0],
            [0.0, 0.4],
        ]
    )
    components = np.array(
        [
            [1.0, 0.0, 0.0],
            [2**-0.5, -(2**-0.5), 0.0],
        ]
    )

    projected = project_localization_to_pca(cell_taper, components)

    np.testing.assert_allclose(projected, [[1.0, 0.2], [0.75, 0.6]])
    np.testing.assert_allclose(
        project_localization_to_pca(cell_taper, -components),
        projected,
    )
    assert np.all((projected >= 0.0) & (projected <= 1.0))


def test_project_localization_to_pca_rejects_incompatible_shapes() -> None:
    with pytest.raises(ValueError, match="cell dimension"):
        project_localization_to_pca(np.ones((3, 2)), np.ones((2, 4)))
