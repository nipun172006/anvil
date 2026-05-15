"""Hessian-aware diagonal equilibration precision agent."""

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


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


class HessianRuizGeometryAgent:
    """Use Ruiz-style diagonal equilibration on the local Hessian."""

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
        self.iters = max(0, _env_int("RUIZ_ITERS", 10))
        self.norm = os.environ.get("RUIZ_NORM", "fro").strip().lower()
        if self.norm not in {"fro", "l1"}:
            self.norm = "fro"
        self.power = _env_float("GEOM_POWER", 1.0)
        self.blend_identity = float(np.clip(_env_float("GEOM_BLEND_IDENTITY", 0.0), 0.0, 1.0))
        self._cache: dict[int, np.ndarray] = {}
        self._usable = self._check_usable()

    def _check_usable(self) -> bool:
        try:
            return np.asarray(self.R, dtype=float).shape == (self.N, self.N)
        except Exception:
            return False

    def _row_norms(self, S: np.ndarray) -> np.ndarray:
        if self.norm == "l1":
            return np.sum(np.abs(S), axis=1)
        return np.sqrt(np.sum(S * S, axis=1))

    def _ruiz_precision(self, H: np.ndarray) -> np.ndarray:
        d = np.ones(self.N, dtype=float)
        for _ in range(self.iters):
            S = (d[:, None] * H) * d[None, :]
            row_norms = np.maximum(self._row_norms(S), self.eps)
            scale = 1.0 / np.sqrt(row_norms)
            d *= scale
            log_d = np.log(np.maximum(d, self.eps))
            d /= np.exp(float(np.mean(log_d)))

        precision = d * d
        precision = precision ** self.power
        if self.blend_identity > 0.0:
            precision = precision ** (1.0 - self.blend_identity)
        return sanitize_precision(precision, self.N)

    def _precision_for_index(self, idx: int) -> np.ndarray:
        if idx in self._cache:
            return self._cache[idx].copy()
        if not self._usable:
            return np.ones(self.N, dtype=float)

        try:
            H = compute_hessian(self.X[idx], self.X, self.R, self.eta, self.beta)
            precision = self._ruiz_precision(H)
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
