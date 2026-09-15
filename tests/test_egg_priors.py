from __future__ import annotations

import importlib

import numpy as np
import pytest

from fmgeo.priors.egg_augment import augment_crops, directional_rotation_allowed
from fmgeo.priors.egg_mps import MPSCapabilityError, generate_mps_realizations
from fmgeo.priors.egg_procedural import generate_procedural_egg


def test_crops_are_deterministic_and_in_bounds() -> None:
    ensemble = np.arange(3 * 7 * 12 * 12).reshape(3, 7, 12, 12)

    first, first_metadata = augment_crops(
        ensemble, crop_shape=(5, 8, 8), count=10, seed=11, rotations=False
    )
    second, second_metadata = augment_crops(
        ensemble, crop_shape=(5, 8, 8), count=10, seed=11, rotations=False
    )

    assert first.shape == (10, 5, 8, 8)
    np.testing.assert_array_equal(first, second)
    assert first_metadata == second_metadata
    assert all(0 <= item["source_index"] < 3 for item in first_metadata)
    assert all(not item["flip_z"] for item in first_metadata)


def test_rotations_are_disabled_for_directional_ensemble() -> None:
    x = np.arange(20, dtype=float)[None, None, :]
    directional = np.repeat(x, 6, axis=1)
    ensemble = np.repeat(directional[None], 8, axis=0)

    assert not directional_rotation_allowed(ensemble, relative_tolerance=0.2)
    _, metadata = augment_crops(
        ensemble,
        crop_shape=(1, 6, 6),
        count=4,
        seed=3,
        rotations=True,
        rotation_variogram_tolerance=0.2,
    )

    assert all(item["rotation_quarters"] == 0 for item in metadata)


def test_missing_mps_dependency_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    original = importlib.import_module

    def missing(name: str) -> object:
        if name in {"geone", "mpslib"}:
            raise ImportError(name)
        return original(name)

    monkeypatch.setattr(importlib, "import_module", missing)

    with pytest.raises(MPSCapabilityError, match="geone or mpslib"):
        generate_mps_realizations(np.ones((2, 3, 4, 5)), count=2, seed=5)


def test_explicit_mps_adapter_is_shape_checked() -> None:
    images = np.ones((2, 3, 4, 5), dtype=np.uint8)

    generated = generate_mps_realizations(
        images,
        count=3,
        seed=5,
        backend=lambda source, count, seed: np.full(
            (count, *source.shape[1:]), seed % 2, dtype=np.uint8
        ),
    )

    assert generated.shape == (3, 3, 4, 5)
    assert np.all(generated == 1)


def test_procedural_channels_are_connected_and_masked() -> None:
    active = np.ones((7, 24, 30), dtype=bool)
    active[:, 0, 0] = False

    first = generate_procedural_egg(active, seed=19, channel_width_cells=3)
    second = generate_procedural_egg(active, seed=19, channel_width_cells=3)

    np.testing.assert_array_equal(first.facies, second.facies)
    assert first.facies.shape == active.shape
    assert first.facies.dtype == np.uint8
    assert np.all(first.facies[~active] == 255)
    assert np.all(first.logk[~active] == 0)
    assert first.has_spanning_channel
