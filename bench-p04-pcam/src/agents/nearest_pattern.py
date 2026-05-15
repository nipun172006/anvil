"""Nearest-pattern agreement precision policy."""

from __future__ import annotations

import numpy as np

from src.utils.distances import euclidean_distances
from src.utils.normalization import sanitize_precision


class NearestPatternPrecisionAgent:
    """Trust dimensions where the query agrees with its nearest stored pattern."""

    def __init__(
        self,
        stored_patterns: np.ndarray,
        model_params: dict | None = None,
        strength: float = 0.85,
        eps: float = 1e-6,
    ):
        self.X = np.asarray(stored_patterns, dtype=float)
        self.N = int(self.X.shape[1])
        self.model_params = model_params or {}
        self.strength = float(strength)
        self.eps = float(eps)
        self.scale = np.maximum(np.std(self.X, axis=0), self.eps)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=float).reshape(-1)
        if q.shape != (self.N,) or self.X.shape[0] == 0:
            return np.ones(self.N, dtype=float)

        distances = euclidean_distances(self.X, q)
        nearest = self.X[int(np.argmin(distances))]
        scaled_diff = np.abs(q - nearest) / (self.scale + self.eps)
        agreement = 1.0 / np.sqrt(self.eps + scaled_diff)
        precision = agreement ** self.strength
        return sanitize_precision(precision, self.N)
