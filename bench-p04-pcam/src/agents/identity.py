"""Identity precision agent."""

from __future__ import annotations

import numpy as np


class IdentityPrecisionAgent:
    """Baseline policy: trust every dimension equally."""

    def __init__(self, stored_patterns: np.ndarray, model_params: dict | None = None):
        self.X = np.asarray(stored_patterns, dtype=float)
        self.N = int(self.X.shape[1])
        self.model_params = model_params or {}

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        return np.ones(self.N, dtype=float)
