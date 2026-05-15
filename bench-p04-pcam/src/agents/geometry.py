"""Geometry-aware precision policy."""

from __future__ import annotations

import os
import sys

import numpy as np

from src.utils.normalization import sanitize_precision


class GeometryPrecisionAgent:
    """Balance per-dimension curvature when model geometry is available."""

    VECTOR_KEYS = (
        "curvature",
        "diag_curvature",
        "hessian_diag",
        "diag_hessian",
        "H_diag",
    )
    MATRIX_KEYS = ("R", "H", "hessian", "A")

    def __init__(
        self,
        stored_patterns: np.ndarray,
        model_params: dict | None = None,
        strength: float = 1.0,
        eps: float = 1e-8,
    ):
        self.X = np.asarray(stored_patterns, dtype=float)
        self.N = int(self.X.shape[1])
        self.model_params = model_params or {}
        self.strength = float(strength)
        self.eps = float(eps)
        self.curvature = self._extract_curvature()
        self.fallback = self.curvature is None
        self.static_precision = self._build_precision()
        self._debug_once()

    def _extract_curvature(self) -> np.ndarray | None:
        for key in self.VECTOR_KEYS:
            value = self.model_params.get(key)
            if value is None:
                continue
            arr = np.asarray(value, dtype=float).reshape(-1)
            if arr.shape == (self.N,) and np.all(np.isfinite(arr)):
                return np.abs(arr) + self.eps

        for key in self.MATRIX_KEYS:
            value = self.model_params.get(key)
            if value is None:
                continue
            mat = np.asarray(value, dtype=float)
            if mat.ndim != 2 or self.N not in mat.shape:
                continue
            if mat.shape == (self.N, self.N) and key.lower().startswith(("h", "a")):
                diag = np.abs(np.diag(mat))
            elif mat.shape[1] == self.N:
                diag = np.einsum("ij,ij->j", mat, mat)
            elif mat.shape[0] == self.N:
                diag = np.einsum("ij,ij->i", mat, mat)
            else:
                continue
            if diag.shape == (self.N,) and np.all(np.isfinite(diag)):
                return np.abs(diag) + self.eps

        return None

    def _build_precision(self) -> np.ndarray:
        if self.curvature is None:
            return np.ones(self.N, dtype=float)
        normalized_curvature = self.curvature / max(float(np.mean(self.curvature)), self.eps)
        precision = (1.0 / (self.eps + normalized_curvature)) ** self.strength
        return sanitize_precision(precision, self.N)

    def _debug_once(self) -> None:
        if os.environ.get("DEBUG_GEOMETRY", "").strip() != "1":
            return

        keys = sorted(str(k) for k in self.model_params.keys())
        r_value = self.model_params.get("R")
        r_shape = tuple(np.asarray(r_value).shape) if r_value is not None else None
        print("[geometry-debug] model_params keys:", keys, file=sys.stderr)
        print("[geometry-debug] R shape:", r_shape, file=sys.stderr)
        print("[geometry-debug] fallback:", self.fallback, file=sys.stderr)

        if self.curvature is None:
            print("[geometry-debug] curvature: None", file=sys.stderr)
        else:
            print(
                "[geometry-debug] curvature min/max/mean/std:",
                f"{float(np.min(self.curvature)):.6g}",
                f"{float(np.max(self.curvature)):.6g}",
                f"{float(np.mean(self.curvature)):.6g}",
                f"{float(np.std(self.curvature)):.6g}",
                file=sys.stderr,
            )
        p = self.static_precision
        print(
            "[geometry-debug] precision min/max/mean/std:",
            f"{float(np.min(p)):.6g}",
            f"{float(np.max(p)):.6g}",
            f"{float(np.mean(p)):.6g}",
            f"{float(np.std(p)):.6g}",
            file=sys.stderr,
        )

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        return self.static_precision.copy()
