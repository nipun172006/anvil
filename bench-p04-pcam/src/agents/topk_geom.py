"""Top-k retrieval signal blended with static geometry balancing."""

from __future__ import annotations

import os

import numpy as np

from src.agents.geometry import GeometryPrecisionAgent
from src.agents.topk_consensus import TopKConsensusPrecisionAgent
from src.utils.normalization import sanitize_precision


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


class TopKGeometryPrecisionAgent:
    """Blend top-k consensus precision with geometry precision.

    This deliberately avoids the magnitude and nearest-pattern components that
    regressed in the first official quick pass.
    """

    def __init__(
        self,
        stored_patterns: np.ndarray,
        model_params: dict | None = None,
        topk_weight: float = 0.85,
        geom_weight: float = 0.15,
    ):
        self.X = np.asarray(stored_patterns, dtype=float)
        self.N = int(self.X.shape[1])
        self.model_params = model_params or {}
        self.topk_weight = _env_float("TOPK_WEIGHT", float(topk_weight))
        self.geom_weight = _env_float("GEOM_WEIGHT", float(geom_weight))
        if not np.isfinite(self.topk_weight):
            self.topk_weight = float(topk_weight)
        if not np.isfinite(self.geom_weight):
            self.geom_weight = float(geom_weight)
        self.topk = TopKConsensusPrecisionAgent(self.X, self.model_params)
        self.geometry = GeometryPrecisionAgent(self.X, self.model_params)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=float).reshape(-1)
        if q.shape != (self.N,):
            return np.ones(self.N, dtype=float)

        topk_precision = sanitize_precision(self.topk.predict_precision(q), self.N)
        geometry_precision = sanitize_precision(
            self.geometry.predict_precision(q), self.N
        )
        precision = (topk_precision ** self.topk_weight) * (
            geometry_precision ** self.geom_weight
        )
        return sanitize_precision(precision, self.N)
