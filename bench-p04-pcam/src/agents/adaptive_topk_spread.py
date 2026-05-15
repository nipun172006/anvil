"""Adaptive regime split between top-k retrieval and spread optimization."""

from __future__ import annotations

import os
import sys

import numpy as np

from src.agents.spread_optimized_geometry import SpreadOptimizedGeometryAgent
from src.agents.topk_consensus import TopKConsensusPrecisionAgent
from src.utils.distances import euclidean_distances
from src.utils.normalization import sanitize_precision


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _optional_env_float(name: str) -> float | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


class AdaptiveTopKSpreadAgent:
    """Use spread-optimized precision near attractors, top-k elsewhere."""

    def __init__(self, stored_patterns: np.ndarray, model_params: dict | None = None):
        self.X = np.asarray(stored_patterns, dtype=float)
        self.K, self.N = self.X.shape
        self.model_params = model_params or {}
        self.topk = TopKConsensusPrecisionAgent(
            self.X,
            self.model_params,
            k=12,
            temp=1.0,
            consensus_weight=0.45,
            distance="euclidean",
        )
        self.spread = SpreadOptimizedGeometryAgent(self.X, self.model_params)
        self.distance_threshold = _env_float("CLEAN_DISTANCE_THRESHOLD", 0.30)
        self.cosine_threshold = _env_float("CLEAN_COSINE_THRESHOLD", 0.98)
        self.detect_mode = os.environ.get("CLEAN_DETECT_MODE", "distance").strip().lower()
        if self.detect_mode not in {"distance", "cosine", "either"}:
            self.detect_mode = "distance"
        self.guard_distance = _optional_env_float("RETRIEVAL_GUARD_DISTANCE")
        self.guard_cosine = _optional_env_float("RETRIEVAL_GUARD_COSINE")
        self.debug = os.environ.get("ADAPTIVE_DEBUG", "").strip() == "1"
        self._debug_count = 0

    def _nearest_stats(self, q: np.ndarray) -> tuple[int, float, float]:
        distances = euclidean_distances(self.X, q)
        idx = int(np.argmin(distances))
        sq_dist = float(distances[idx])
        q_norm = float(np.linalg.norm(q))
        x_norm = float(np.linalg.norm(self.X[idx]))
        cosine = 0.0
        if q_norm > 1e-12 and x_norm > 1e-12:
            cosine = float((self.X[idx] @ q) / (x_norm * q_norm))
        return idx, sq_dist, cosine

    def _is_clean(self, sq_dist: float, cosine: float) -> bool:
        distance_clean = sq_dist <= self.distance_threshold
        cosine_clean = cosine >= self.cosine_threshold
        if self.detect_mode == "cosine":
            return cosine_clean
        if self.detect_mode == "either":
            return distance_clean or cosine_clean
        return distance_clean

    def _guard_to_spread(self, sq_dist: float, cosine: float) -> bool:
        distance_guarded = (
            self.guard_distance is not None and sq_dist <= self.guard_distance
        )
        cosine_guarded = self.guard_cosine is not None and cosine >= self.guard_cosine
        return distance_guarded or cosine_guarded

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=float).reshape(-1)
        if q.shape != (self.N,) or self.K == 0:
            return np.ones(self.N, dtype=float)

        idx, sq_dist, cosine = self._nearest_stats(q)
        clean = self._is_clean(sq_dist, cosine) or self._guard_to_spread(sq_dist, cosine)
        if self.debug and self._debug_count < 12:
            print(
                "[adaptive-debug]",
                f"idx={idx}",
                f"sq_dist={sq_dist:.6g}",
                f"cosine={cosine:.6g}",
                f"mode={'spread' if clean else 'topk'}",
                file=sys.stderr,
            )
            self._debug_count += 1

        if clean:
            return sanitize_precision(self.spread.predict_precision(q), self.N)
        return sanitize_precision(self.topk.predict_precision(q), self.N)
