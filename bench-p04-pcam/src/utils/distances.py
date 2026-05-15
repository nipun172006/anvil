"""Distance and normalization utilities."""

from __future__ import annotations

import numpy as np


def safe_l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize an array while avoiding division by zero."""
    arr = np.asarray(x, dtype=float)
    norm = np.linalg.norm(arr, axis=axis, keepdims=True)
    return arr / np.maximum(norm, eps)


def euclidean_distances(patterns: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Squared Euclidean distances from ``query`` to every stored pattern."""
    x = np.asarray(patterns, dtype=float)
    q = np.asarray(query, dtype=float).reshape(-1)
    if x.ndim != 2 or q.shape != (x.shape[1],):
        return np.full(x.shape[0] if x.ndim == 2 else 0, np.inf, dtype=float)
    diff = x - q[None, :]
    return np.einsum("ij,ij->i", diff, diff)


def cosine_distances(patterns: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine distances from ``query`` to every stored pattern."""
    x = np.asarray(patterns, dtype=float)
    q = np.asarray(query, dtype=float).reshape(-1)
    if x.ndim != 2 or q.shape != (x.shape[1],):
        return np.full(x.shape[0] if x.ndim == 2 else 0, np.inf, dtype=float)

    x_norm = safe_l2_normalize(x, axis=1)
    q_norm = safe_l2_normalize(q, axis=0)
    similarity = x_norm @ q_norm
    return 1.0 - np.clip(similarity, -1.0, 1.0)
