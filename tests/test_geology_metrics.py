from __future__ import annotations

import numpy as np

from fmgeo.metrics.geology import (
    connected_component_sizes,
    experimental_variogram,
    facies_fraction_by_layer,
    well_column_connected,
    well_column_connectivity_probability,
    well_pair_connected,
    well_pair_connectivity_probability,
)


def test_facies_fraction_respects_layers_and_active_mask() -> None:
    facies = np.array([[[1, 0], [1, 1]], [[0, 0], [1, 1]]], dtype=np.uint8)
    active = np.array([[[1, 1], [1, 0]], [[1, 0], [1, 1]]], dtype=bool)

    fractions = facies_fraction_by_layer(facies, facies=1, active_mask=active)

    np.testing.assert_allclose(fractions, [2 / 3, 2 / 3])


def test_variogram_of_linear_field_has_analytical_values() -> None:
    field = np.arange(6, dtype=float)[None, None, :]

    variogram = experimental_variogram(field, axis=2, max_lag=3)

    np.testing.assert_allclose(variogram, [0.0, 0.5, 2.0, 4.5])


def test_component_sizes_use_face_connectivity() -> None:
    sand = np.zeros((2, 3, 3), dtype=bool)
    sand[0, 0, 0] = True
    sand[0, 0, 1] = True
    sand[1, 2, 2] = True

    assert connected_component_sizes(sand).tolist() == [1, 2]


def test_well_pair_connectivity_and_probability() -> None:
    connected = np.zeros((1, 3, 3), dtype=bool)
    connected[0, 1, :] = True
    disconnected = connected.copy()
    disconnected[0, 1, 1] = False
    first = (0, 1, 0)
    second = (0, 1, 2)

    assert well_pair_connected(connected, first, second)
    assert not well_pair_connected(disconnected, first, second)
    assert well_pair_connectivity_probability(
        np.stack([connected, disconnected]), first, second
    ) == 0.5


def test_well_column_connectivity_accepts_connection_in_any_layer() -> None:
    sand = np.zeros((3, 4, 5), dtype=bool)
    sand[1, 2, 1:5] = True

    assert well_column_connected(sand, (2, 1), (2, 4))
    assert not well_column_connected(sand, (0, 0), (2, 4))
    ensemble = np.stack([sand, np.zeros_like(sand)])
    assert well_column_connectivity_probability(ensemble, (2, 1), (2, 4)) == 0.5
