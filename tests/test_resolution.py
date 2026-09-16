from __future__ import annotations

import numpy as np

from fmgeo.resolution import average_pool_horizontal, scale_correlation_lengths


def test_horizontal_pooling_averages_only_active_cells_deterministically() -> None:
    fields = np.asarray(
        [[[[1.0, 3.0, 10.0, 14.0], [5.0, 7.0, 18.0, 22.0]]]], dtype=np.float32
    )
    active = np.asarray([[[True, True, True, False], [True, True, False, False]]])

    pooled, pooled_active = average_pool_horizontal(fields, active_mask=active, factor=2)

    np.testing.assert_array_equal(pooled_active, [[[True, True]]])
    np.testing.assert_allclose(pooled, [[[[4.0, 10.0]]]])
    assert pooled.dtype == fields.dtype


def test_correlation_lengths_preserve_physical_scale_across_resolution() -> None:
    scaled = scale_correlation_lengths(
        (1.0, 4.0, 8.0),
        reference_shape=(7, 60, 60),
        target_shape=(7, 30, 30),
    )

    assert scaled == (1.0, 2.0, 4.0)
