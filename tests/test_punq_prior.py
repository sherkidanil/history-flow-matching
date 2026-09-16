from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from fmgeo.priors.punq_objectbased import (
    ChannelGeometry,
    FaciesProperty,
    HardDatum,
    LayerPrior,
    PunqPriorConfig,
    generate_punq_realization,
    load_punq_prior_config,
    validate_prior_against_truth,
    write_punq_prior_hdf5,
)


def property_model(mean: float) -> FaciesProperty:
    return FaciesProperty(
        logk_mean=mean,
        logk_std=0.2,
        porosity_intercept=0.05,
        porosity_slope=0.02,
        porosity_bounds=(0.01, 0.30),
    )


def small_config() -> PunqPriorConfig:
    geometry = ChannelGeometry(
        width_mean_m=2.0,
        width_std_m=0.1,
        amplitude_range_m=(0.5, 2.0),
        wavelength_range_m=(5.0, 12.0),
        azimuth_range_degrees=(110.0, 170.0),
        net_to_gross_width_fraction=1.0,
    )
    layers = []
    for layer in range(5):
        if layer in (0, 2, 4):
            layers.append(
                LayerPrior(
                    kind="channel",
                    background=property_model(1.0),
                    sand=property_model(4.0),
                    geometry=geometry,
                    vertical_ratio=0.25,
                    property_corr_len_cells=(1.0, 1.5),
                )
            )
        else:
            layers.append(
                LayerPrior(
                    kind="homogeneous",
                    background=property_model(2.0),
                    vertical_ratio=0.15,
                    property_corr_len_cells=(1.0, 1.0),
                )
            )
    hard_data = tuple(
        HardDatum(z=z, y=5, x=5, facies=1 if z in (0, 2, 4) else 0)
        for z in range(5)
    )
    return PunqPriorConfig(
        shape=(5, 12, 12),
        cell_size_m=(1.0, 1.0),
        channels_per_channel_layer=3,
        layers=tuple(layers),
        hard_data=hard_data,
    )


def test_realization_is_deterministic_with_exact_shapes_and_dtypes() -> None:
    config = small_config()
    active = np.ones(config.shape, dtype=bool)

    first = generate_punq_realization(config, active, seed=17)
    second = generate_punq_realization(config, active, seed=17)

    assert first.logk.shape == config.shape
    assert first.logk.dtype == np.float32
    assert first.facies.dtype == np.uint8
    assert first.porosity.dtype == np.float32
    np.testing.assert_array_equal(first.facies, second.facies)
    np.testing.assert_array_equal(first.logk, second.logk)


def test_channel_counts_homogeneous_layers_and_hard_data() -> None:
    config = small_config()
    active = np.ones(config.shape, dtype=bool)

    result = generate_punq_realization(config, active, seed=29)

    assert {layer: len(channels) for layer, channels in result.channels.items()} == {
        0: 3,
        2: 3,
        4: 3,
    }
    assert np.all(result.facies[1] == 0)
    assert np.all(result.facies[3] == 0)
    for datum in config.hard_data:
        assert result.facies[datum.z, datum.y, datum.x] == datum.facies


def test_permeability_anisotropy_and_porosity_are_derived() -> None:
    config = small_config()
    active = np.ones(config.shape, dtype=bool)

    result = generate_punq_realization(config, active, seed=31)

    np.testing.assert_allclose(result.permx, result.permy)
    for z, layer in enumerate(config.layers):
        np.testing.assert_allclose(
            result.permz[z], result.permx[z] * layer.vertical_ratio, rtol=1e-6
        )
        expected = np.empty(result.logk[z].shape)
        for facies, properties in ((0, layer.background), (1, layer.sand)):
            if properties is None:
                continue
            selected = result.facies[z] == facies
            expected[selected] = np.clip(
                properties.porosity_intercept
                + properties.porosity_slope * result.logk[z, selected],
                *properties.porosity_bounds,
            )
        np.testing.assert_allclose(result.porosity[z], expected, rtol=1e-6)


def test_inactive_cells_are_masked() -> None:
    config = small_config()
    active = np.ones(config.shape, dtype=bool)
    active[:, 0, 0] = False

    result = generate_punq_realization(config, active, seed=41)

    assert np.all(result.facies[:, 0, 0] == 255)
    assert np.all(result.logk[:, 0, 0] == 0)
    assert np.all(result.porosity[:, 0, 0] == 0)


def test_project_config_contains_all_thirty_conditioning_cells() -> None:
    config_path = Path(__file__).parents[1] / "configs/punq/prior.yaml"

    config = load_punq_prior_config(config_path)

    assert config.shape == (5, 28, 19)
    assert len(config.hard_data) == 30
    assert len({(item.z, item.y, item.x) for item in config.hard_data}) == 30


def test_hdf5_writer_stores_only_required_ensemble_fields(tmp_path: Path) -> None:
    config = small_config()
    active = np.ones(config.shape, dtype=bool)
    destination = tmp_path / "prior.h5"

    metadata = write_punq_prior_hdf5(
        destination, config=config, active_mask=active, count=3, seed=53
    )

    with h5py.File(destination) as handle:
        assert set(handle.keys()) == {"active_mask", "facies", "logk", "sample_seeds"}
        assert handle["logk"].shape == (3, *config.shape)
        assert handle["logk"].dtype == np.dtype("float32")
        assert handle["facies"].dtype == np.dtype("uint8")
        assert handle["active_mask"].shape == config.shape
    assert metadata["shape"] == (3, *config.shape)
    assert metadata["dtype"] == "float32"


def test_validation_accepts_an_ensemble_matching_truth() -> None:
    config = small_config()
    active = np.ones(config.shape, dtype=bool)
    rng = np.random.default_rng(71)
    truth_logk = rng.normal(size=config.shape)
    truth_facies = rng.integers(0, 2, size=config.shape, dtype=np.uint8)
    truth_facies[1] = 0
    truth_facies[3] = 0
    truth_poro = np.where(truth_facies == 1, 0.25, 0.05)
    logk_ensemble = np.repeat(truth_logk[None], 8, axis=0)
    facies_ensemble = np.repeat(truth_facies[None], 8, axis=0)

    report = validate_prior_against_truth(
        config=config,
        active_mask=active,
        facies_ensemble=facies_ensemble,
        logk_ensemble=logk_ensemble,
        truth_logk=truth_logk,
        truth_porosity=truth_poro,
    )

    assert report["accepted"] is True
    assert report["histogram_normalized_wasserstein"] == 0.0
    assert report["variogram_envelope_coverage"] == 1.0
