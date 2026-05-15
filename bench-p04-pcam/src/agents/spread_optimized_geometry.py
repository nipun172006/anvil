"""Precomputed spread-optimized precision per stored pattern."""

from __future__ import annotations

import os
import sys

import numpy as np

from src.utils.distances import euclidean_distances
from src.utils.hessian import compute_hessian
from src.utils.normalization import sanitize_precision
from src.utils.spread_objective import (
    optimize_precision_for_hessian,
    stable_condition_number,
)


class SpreadOptimizedGeometryAgent:
    """Nearest-attractor precision table optimized for Hessian spread."""

    def __init__(self, stored_patterns: np.ndarray, model_params: dict | None = None):
        self.X = np.asarray(stored_patterns, dtype=float)
        self.K, self.N = self.X.shape
        self.model_params = model_params or {}
        self.R = self.model_params.get("R")
        self.eta = float(self.model_params.get("eta", 0.5))
        self.beta = float(self.model_params.get("beta", 8.0))
        self.budget = int(os.environ.get("SPREAD_OPT_BUDGET", 80))
        self.seed = int(os.environ.get("SPREAD_OPT_SEED", 123))
        self._usable = self._check_usable()
        self.precisions = self._precompute()

    def _check_usable(self) -> bool:
        try:
            return np.asarray(self.R, dtype=float).shape == (self.N, self.N)
        except Exception:
            return False

    def _precompute(self) -> np.ndarray:
        if not self._usable:
            return np.ones((self.K, self.N), dtype=float)

        precisions = np.ones((self.K, self.N), dtype=float)
        debug_rows: list[tuple[int, float, float]] = []
        for idx, pattern in enumerate(self.X):
            try:
                H = compute_hessian(pattern, self.X, self.R, self.eta, self.beta)
                pi = optimize_precision_for_hessian(
                    H,
                    self.N,
                    seed=self.seed + idx,
                    budget=self.budget,
                )
                pi = sanitize_precision(pi, self.N)
                precisions[idx] = pi
                if len(debug_rows) < 4:
                    base = stable_condition_number(H, np.ones(self.N, dtype=float))
                    opt = stable_condition_number(H, pi)
                    debug_rows.append((idx, base, opt))
            except Exception:
                precisions[idx] = np.ones(self.N, dtype=float)

        if os.environ.get("DEBUG_SPREAD_OPT", "").strip() == "1":
            print(
                "[spread-opt-debug] usable:",
                self._usable,
                "K:",
                self.K,
                "N:",
                self.N,
                "budget:",
                self.budget,
                file=sys.stderr,
            )
            for idx, base, opt in debug_rows:
                ratio = base / opt if opt > 0 and np.isfinite(opt) else 0.0
                print(
                    "[spread-opt-debug]",
                    f"idx={idx}",
                    f"base={base:.6g}",
                    f"opt={opt:.6g}",
                    f"ratio={ratio:.4g}x",
                    file=sys.stderr,
                )
            p = precisions
            print(
                "[spread-opt-debug] precision min/max/mean/std:",
                f"{float(np.min(p)):.6g}",
                f"{float(np.max(p)):.6g}",
                f"{float(np.mean(p)):.6g}",
                f"{float(np.std(p)):.6g}",
                file=sys.stderr,
            )

        return precisions

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        q = np.asarray(corrupted_query, dtype=float).reshape(-1)
        if q.shape != (self.N,) or self.K == 0:
            return np.ones(self.N, dtype=float)
        idx = int(np.argmin(euclidean_distances(self.X, q)))
        return self.precisions[idx].copy()
