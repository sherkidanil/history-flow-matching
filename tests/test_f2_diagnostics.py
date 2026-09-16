from __future__ import annotations

import numpy as np
import pytest

from fmgeo.inverse.diagnostics import (
    midpoint_nonlinearity,
    relative_frobenius_shift,
    sample_member_pairs,
)


def test_sample_member_pairs_is_seeded_unique_and_excludes_self_pairs() -> None:
    left = sample_member_pairs(5, 6, rng=np.random.default_rng(7))
    right = sample_member_pairs(5, 6, rng=np.random.default_rng(7))

    np.testing.assert_array_equal(left, right)
    assert left.shape == (6, 2)
    assert np.all(left[:, 0] < left[:, 1])
    assert len({tuple(pair) for pair in left}) == 6


def test_sample_member_pairs_rejects_more_than_unique_pair_count() -> None:
    with pytest.raises(ValueError, match="unique member pairs"):
        sample_member_pairs(3, 4, rng=np.random.default_rng(1))


def test_midpoint_nonlinearity_is_zero_for_linear_decoder() -> None:
    parameters = np.array([[0.0], [2.0], [4.0]])
    decoded = 2.0 * parameters
    pairs = np.array([[0, 1], [1, 2]])

    errors = midpoint_nonlinearity(
        parameters,
        decode=lambda values: 2.0 * values,
        pairs=pairs,
        decoded_parameters=decoded,
    )

    np.testing.assert_allclose(errors, 0.0, atol=1e-15)


def test_midpoint_nonlinearity_measures_nonlinear_decoder() -> None:
    parameters = np.array([[0.0], [2.0], [4.0]])
    pairs = np.array([[0, 1], [1, 2]])

    errors = midpoint_nonlinearity(
        parameters,
        decode=np.square,
        pairs=pairs,
        decoded_parameters=np.square(parameters),
    )

    np.testing.assert_allclose(errors, [0.5, 0.1])


def test_relative_frobenius_shift_is_scale_normalized() -> None:
    reference = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert relative_frobenius_shift(reference, 2.0 * reference) == pytest.approx(1.0)


def test_relative_frobenius_shift_rejects_zero_reference() -> None:
    with pytest.raises(ValueError, match="positive norm"):
        relative_frobenius_shift(np.zeros((2, 2)), np.ones((2, 2)))
