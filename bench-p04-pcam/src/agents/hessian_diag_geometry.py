"""Hessian-diagonal geometry precision agent."""

from __future__ import annotations

import os

import numpy as np

from src.utils.distances import euclidean_distances
from src.utils.hessian import compute_hessian
from src.utils.normalization import sanitize_precision


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


class HessianDiagonalGeometryAgent:
    """Use inverse local Hessian diagonal as a per-dimension precision proxy."""

    def __init__(
        self,
        stored_patterns: np.ndarray,
        model_params: dict | None = None,
        eps: float = 1e-8,
    ):
        self.X = np.asarray(stored_patterns, dtype=float)
        self.N = int(self.X.shape[1])
        self.model_params = model_params or {}
        self.R = self.model_params.get("R")
        self.eta = float(self.model_params.get("eta", 0.5))
        self.beta = float(self.model_params.get("beta", 8.0))
        self.eps = float(eps)
        self.power = _env_float("GEOM_POWER", 1.0)
        self._cache: dict[int, np.ndarray] = {}
        self._usable = self._check_usable()

    def _check_usable(self) -> bool:
        try:
            return np.asarray(self.R, dtype=float).shape == (self.N, self.N)
        except Exception:
            return False

    def _precision_for_index(self, idx: int) -> np.ndarray:
        if idx in self._cache:
            return self._cache[idx].copy()
        if not self._usable:
            return np.ones(self.N, dtype=float)

        try:
            H = compute_hessian(self.X[idx], self.X, self.R, self.eta, self.beta)
            diag = np.abs(np.diag(H))
            precision = 1.0 / (self.eps + diag)
            precision = precision ** self.power
            precision = sanitize_precision(precision, self.N)
        except Exception:
            precision = np.ones(self.N, dtype=float)

        self._cache[idx] = precision
        return precision.copy()

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=float).reshape(-1)
        if q.shape != (self.N,) or self.X.shape[0] == 0:
            return np.ones(self.N, dtype=float)
        idx = int(np.argmin(euclidean_distances(self.X, q)))
        return self._precision_for_index(idx)
