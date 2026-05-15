"""Magnitude-based precision policy."""

from __future__ import annotations

import numpy as np

from src.utils.normalization import sanitize_precision


class MagnitudePrecisionAgent:
    """Trust dimensions whose observed query magnitude looks strong.

    This is useful for masked inputs where unreliable dimensions tend to be
    driven toward zero.
    """

    def __init__(
        self,
        stored_patterns: np.ndarray,
        model_params: dict | None = None,
        strength: float = 0.75,
        eps: float = 1e-8,
    ):
        self.X = np.asarray(stored_patterns, dtype=float)
        self.N = int(self.X.shape[1])
        self.model_params = model_params or {}
        self.strength = float(strength)
        self.eps = float(eps)

        abs_x = np.abs(self.X)
        per_dim = np.median(abs_x, axis=0)
        global_scale = float(np.median(abs_x)) if abs_x.size else 1.0
        self.scale = np.maximum(per_dim, global_scale * 0.25 + self.eps)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=float).reshape(-1)
        if q.shape != (self.N,):
            return np.ones(self.N, dtype=float)

        reliability = np.abs(q) / (self.scale + self.eps)
        reliability = np.clip(reliability, 0.0, 3.0)
        centered = reliability - float(np.mean(reliability))
        precision = np.exp(self.strength * np.tanh(centered))
        return sanitize_precision(precision, self.N)
