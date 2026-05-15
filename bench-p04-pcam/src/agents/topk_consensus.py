"""Top-k candidate consensus precision policy."""

from __future__ import annotations

import os

import numpy as np

from src.utils.distances import cosine_distances, euclidean_distances
from src.utils.normalization import sanitize_precision


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


class TopKConsensusPrecisionAgent:
    """Use the local candidate set to infer reliable dimensions."""

    def __init__(
        self,
        stored_patterns: np.ndarray,
        model_params: dict | None = None,
        k: int = 5,
        agreement_weight: float = 0.55,
        consensus_weight: float = 0.45,
        temp: float = 1.0,
        power: float = 1.0,
        distance: str = "euclidean",
        eps: float = 1e-6,
    ):
        self.X = np.asarray(stored_patterns, dtype=float)
        self.N = int(self.X.shape[1])
        self.model_params = model_params or {}
        self.k = max(1, _env_int("TOPK_K", int(k)))
        self.agreement_weight = _env_float(
            "TOPK_AGREEMENT_WEIGHT", float(agreement_weight)
        )
        self.consensus_weight = _env_float(
            "TOPK_CONSENSUS_WEIGHT", float(consensus_weight)
        )
        self.temp = max(_env_float("TOPK_TEMP", float(temp)), eps)
        self.power = _env_float("TOPK_POWER", float(power))
        self.distance = os.environ.get("TOPK_DISTANCE", distance).strip().lower()
        if self.distance not in {"euclidean", "cosine", "dot"}:
            self.distance = "euclidean"
        self.eps = float(eps)
        self.scale = np.maximum(np.std(self.X, axis=0), self.eps)

    def _distances(self, query: np.ndarray) -> np.ndarray:
        if self.distance == "cosine":
            return cosine_distances(self.X, query)
        if self.distance == "dot":
            return -(self.X @ query)
        return euclidean_distances(self.X, query)

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=float).reshape(-1)
        if q.shape != (self.N,) or self.X.shape[0] == 0:
            return np.ones(self.N, dtype=float)

        k = min(self.k, self.X.shape[0])
        distances = self._distances(q)
        top_idx = np.argpartition(distances, kth=k - 1)[:k]
        top = self.X[top_idx]
        top_distances = distances[top_idx]
        local_scale = (
            float(np.median(top_distances - np.min(top_distances))) * self.temp
            + self.eps
        )
        weights = np.exp(-(top_distances - np.min(top_distances)) / local_scale)
        weights = weights / max(float(np.sum(weights)), self.eps)

        center = weights @ top
        variance = weights @ ((top - center[None, :]) ** 2)
        scaled_diff = np.abs(q - center) / (self.scale + self.eps)
        scaled_var = variance / (self.scale**2 + self.eps)

        agreement = 1.0 / np.sqrt(0.25 + scaled_diff)
        consensus = 1.0 / np.sqrt(1.0 + scaled_var)
        precision = (agreement ** self.agreement_weight) * (
            consensus ** self.consensus_weight
        )
        precision = precision ** self.power
        return sanitize_precision(precision, self.N)
