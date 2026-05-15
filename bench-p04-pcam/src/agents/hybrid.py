"""Hybrid precision policy combining query confidence and geometry."""

from __future__ import annotations

import numpy as np

from src.agents.geometry import GeometryPrecisionAgent
from src.agents.magnitude import MagnitudePrecisionAgent
from src.agents.nearest_pattern import NearestPatternPrecisionAgent
from src.agents.topk_consensus import TopKConsensusPrecisionAgent
from src.utils.distances import euclidean_distances
from src.utils.normalization import sanitize_precision


class HybridPrecisionAgent:
    """Current main candidate for the official adapter.

    The policy uses a multiplicative blend so independent confidence signals
    can reinforce each other while every component remains clipped and mean-one.
    """

    TOPK_WEIGHT = 0.10
    NEAREST_WEIGHT = 0.05
    MAGNITUDE_WEIGHT = 0.80
    GEOMETRY_WEIGHT = 0.05

    CLEAN_TOPK_WEIGHT = 0.05
    CLEAN_NEAREST_WEIGHT = 0.05
    CLEAN_MAGNITUDE_WEIGHT = 0.10
    CLEAN_GEOMETRY_WEIGHT = 0.80

    def __init__(
        self,
        stored_patterns: np.ndarray,
        model_params: dict | None = None,
        topk_weight: float = TOPK_WEIGHT,
        nearest_weight: float = NEAREST_WEIGHT,
        magnitude_weight: float = MAGNITUDE_WEIGHT,
        geometry_weight: float = GEOMETRY_WEIGHT,
        clean_gate_scale: float = 0.05,
    ):
        self.X = np.asarray(stored_patterns, dtype=float)
        self.N = int(self.X.shape[1])
        self.model_params = model_params or {}

        self.corrupt_weights = np.asarray(
            [topk_weight, nearest_weight, magnitude_weight, geometry_weight],
            dtype=float,
        )
        invalid_weights = (
            not np.all(np.isfinite(self.corrupt_weights))
            or np.sum(self.corrupt_weights) <= 0.0
        )
        if invalid_weights:
            self.corrupt_weights = np.asarray(
                [
                    self.TOPK_WEIGHT,
                    self.NEAREST_WEIGHT,
                    self.MAGNITUDE_WEIGHT,
                    self.GEOMETRY_WEIGHT,
                ],
                dtype=float,
            )
        self.corrupt_weights = self.corrupt_weights / np.sum(self.corrupt_weights)
        self.clean_weights = np.asarray(
            [
                self.CLEAN_TOPK_WEIGHT,
                self.CLEAN_NEAREST_WEIGHT,
                self.CLEAN_MAGNITUDE_WEIGHT,
                self.CLEAN_GEOMETRY_WEIGHT,
            ],
            dtype=float,
        )
        self.clean_weights = self.clean_weights / np.sum(self.clean_weights)
        self.clean_gate_scale = max(float(clean_gate_scale), 1e-8)
        self.distance_scale = max(float(np.mean(self.X**2)), 1e-8)

        self.topk = TopKConsensusPrecisionAgent(self.X, self.model_params)
        self.nearest = NearestPatternPrecisionAgent(self.X, self.model_params)
        self.magnitude = MagnitudePrecisionAgent(self.X, self.model_params)
        self.geometry = GeometryPrecisionAgent(self.X, self.model_params)

    def _query_weights(self, query: np.ndarray) -> np.ndarray:
        """Blend retrieval-oriented and geometry-oriented weights.

        Exact or nearly exact stored patterns get a high clean gate, which is
        useful for local curvature checks. Corrupted queries get the retrieval
        weights, where magnitude is the most reliable synthetic corruption cue.
        """
        distances = euclidean_distances(self.X, query)
        nearest_mse = float(np.min(distances)) / (self.N * self.distance_scale)
        clean_gate = float(np.exp(-nearest_mse / self.clean_gate_scale))
        clean_gate = float(np.clip(clean_gate, 0.0, 1.0))
        weights = (
            (1.0 - clean_gate) * self.corrupt_weights
            + clean_gate * self.clean_weights
        )
        return weights / max(float(np.sum(weights)), 1e-12)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=float).reshape(-1)
        if q.shape != (self.N,):
            return np.ones(self.N, dtype=float)

        parts = [
            self.topk.predict_precision(q),
            self.nearest.predict_precision(q),
            self.magnitude.predict_precision(q),
            self.geometry.predict_precision(q),
        ]
        safe_parts = [sanitize_precision(part, self.N) for part in parts]
        log_precision = np.zeros(self.N, dtype=float)
        weights = self._query_weights(q)
        for weight, part in zip(weights, safe_parts):
            log_precision += float(weight) * np.log(np.maximum(part, 1e-8))

        precision = np.exp(log_precision)
        return sanitize_precision(precision, self.N)
