"""Principal-component baseline parameterization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class PCAParameterization:
    """Centered truncated-SVD mapping between fields and PCA scores."""

    mean: NDArray[np.float64]
    components: NDArray[np.float64]
    original_shape: tuple[int, ...]
    explained_variance_fraction: float

    @property
    def rank(self) -> int:
        return self.components.shape[0]

    @classmethod
    def fit(
        cls,
        ensemble: ArrayLike,
        *,
        variance_fraction: float = 0.95,
        max_rank: int | None = None,
    ) -> Self:
        values = np.asarray(ensemble, dtype=np.float64)
        if values.ndim < 2 or values.shape[0] < 2:
            raise ValueError("ensemble must contain at least two samples")
        if not 0.0 < variance_fraction <= 1.0:
            raise ValueError("variance_fraction must be in (0, 1]")
        if max_rank is not None and max_rank < 1:
            raise ValueError("max_rank must be positive")
        sample_shape = values.shape[1:]
        flattened = values.reshape(values.shape[0], -1)
        mean = flattened.mean(axis=0)
        _, singular_values, right_vectors = np.linalg.svd(flattened - mean, full_matrices=False)
        variances = singular_values**2
        tolerance = np.finfo(np.float64).eps * max(flattened.shape) * singular_values[0]
        numerical_rank = int(np.count_nonzero(singular_values > tolerance))
        if numerical_rank == 0:
            raise ValueError("cannot fit PCA to a constant ensemble")
        cumulative = np.cumsum(variances[:numerical_rank]) / variances[:numerical_rank].sum()
        rank = min(
            int(np.searchsorted(cumulative, variance_fraction, side="left")) + 1,
            numerical_rank,
        )
        if max_rank is not None:
            rank = min(rank, max_rank)
        return cls(
            mean=mean,
            components=np.asarray(right_vectors[:rank], dtype=np.float64),
            original_shape=sample_shape,
            explained_variance_fraction=float(cumulative[rank - 1]),
        )

    def encode(self, fields: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(fields, dtype=np.float64)
        if values.shape[1:] != self.original_shape:
            raise ValueError(f"expected sample shape {self.original_shape}, got {values.shape[1:]}")
        flattened = values.reshape(values.shape[0], -1)
        return np.asarray((flattened - self.mean) @ self.components.T, dtype=np.float64)

    def decode(self, scores: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.rank:
            raise ValueError(f"expected scores with shape (sample, {self.rank})")
        flattened = values @ self.components + self.mean
        return np.asarray(
            flattened.reshape(values.shape[0], *self.original_shape), dtype=np.float64
        )
