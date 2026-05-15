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
        self.retrieval_policy = os.environ.get("RETRIEVAL_POLICY", "fixed").strip().lower()
        if self.retrieval_policy not in {"fixed", "adaptive_margin"}:
            self.retrieval_policy = "fixed"
        self.margin_threshold = max(_env_float("MARGIN_THRESHOLD", 0.03), 0.0)
        self.ratio_threshold = max(_env_float("RATIO_THRESHOLD", 1.08), 1.0)
        self.distance_threshold = max(_env_float("DISTANCE_THRESHOLD_RETRIEVAL", 0.95), 0.0)
        self.k_ambiguous = max(1, _env_int("ADAPTIVE_K_AMBIG", 16))
        self.k_noisy = max(1, _env_int("ADAPTIVE_K_NOISY", 14))
        self.temp_ambiguous = max(_env_float("ADAPTIVE_TEMP_AMBIG", 1.2), eps)
        self.temp_noisy = max(
            _env_float("ADAPTIVE_TEMP_NOISY", self.temp_ambiguous),
            eps,
        )

    def _distances(self, query: np.ndarray) -> np.ndarray:
        if self.distance == "cosine":
            return cosine_distances(self.X, query)
        if self.distance == "dot":
            return -(self.X @ query)
        return euclidean_distances(self.X, query)

    def _policy_params(self, distances: np.ndarray) -> tuple[int, float, float]:
        if self.retrieval_policy != "adaptive_margin" or distances.size < 2:
            return self.k, self.temp, self.consensus_weight

        first_two = np.partition(distances, kth=1)[:2]
        first_two.sort()
        d1 = float(first_two[0])
        d2 = float(first_two[1])
        margin = d2 - d1
        ratio = d2 / (d1 + self.eps)

        if margin <= self.margin_threshold or ratio <= self.ratio_threshold:
            return self.k_ambiguous, self.temp_ambiguous, self.consensus_weight
        if d1 >= self.distance_threshold:
            return self.k_noisy, self.temp_noisy, self.consensus_weight
        return self.k, self.temp, self.consensus_weight

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=float).reshape(-1)
        if q.shape != (self.N,) or self.X.shape[0] == 0:
            return np.ones(self.N, dtype=float)

        distances = self._distances(q)
        k, temp, consensus_weight = self._policy_params(distances)
        k = min(max(1, int(k)), self.X.shape[0])
        top_idx = np.argpartition(distances, kth=k - 1)[:k]
        top = self.X[top_idx]
        top_distances = distances[top_idx]
        local_scale = (
            float(np.median(top_distances - np.min(top_distances))) * temp
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
            consensus ** consensus_weight
        )
        precision = precision ** self.power
        return sanitize_precision(precision, self.N)
