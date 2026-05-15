"""Precision-vector sanitation helpers."""

from __future__ import annotations

import numpy as np


MIN_PRECISION = 0.1
MAX_PRECISION = 10.0


def _ones(n: int) -> np.ndarray:
    return np.ones(int(n), dtype=float)


def _project_clipped_mean_one(
    values: np.ndarray,
    lo: float = MIN_PRECISION,
    hi: float = MAX_PRECISION,
    target_mean: float = 1.0,
) -> np.ndarray:
    """Scale ``values`` so clipped output has mean approximately one.

    For positive vectors this solves for ``s`` in
    ``mean(clip(values * s, lo, hi)) = target_mean`` by bisection. Since
    ``lo <= target_mean <= hi``, a feasible clipped solution always exists.
    """
    base = np.clip(np.asarray(values, dtype=float), lo, hi)
    if base.ndim != 1 or base.size == 0 or not np.all(np.isfinite(base)):
        return base

    low = 0.0
    high = 1.0
    for _ in range(64):
        if np.mean(np.clip(base * high, lo, hi)) >= target_mean:
            break
        high *= 2.0

    for _ in range(64):
        mid = 0.5 * (low + high)
        mean = float(np.mean(np.clip(base * mid, lo, hi)))
        if mean < target_mean:
            low = mid
        else:
            high = mid

    return np.clip(base * high, lo, hi)


def sanitize_precision(p: np.ndarray, n: int) -> np.ndarray:
    """Return a finite, positive, clipped, mean-one precision vector.

    Invalid inputs, wrong shapes, all-nonfinite vectors, or degenerate means
    safely become the identity precision vector.
    """
    try:
        n = int(n)
        if n <= 0:
            return np.ones(0, dtype=float)

        arr = np.asarray(p, dtype=float).reshape(-1)
        if arr.shape != (n,):
            return _ones(n)

        arr = np.nan_to_num(
            arr,
            nan=1.0,
            posinf=MAX_PRECISION,
            neginf=MIN_PRECISION,
        )
        if not np.all(np.isfinite(arr)):
            return _ones(n)

        arr = np.abs(arr)
        arr = np.where(arr > 0.0, arr, MIN_PRECISION)
        arr = np.clip(arr, MIN_PRECISION, MAX_PRECISION)
        if not np.any(arr > 0.0):
            return _ones(n)

        arr = _project_clipped_mean_one(arr)
        if arr.shape != (n,) or not np.all(np.isfinite(arr)):
            return _ones(n)

        mean = float(np.mean(arr))
        if mean <= 0.0 or not np.isfinite(mean):
            return _ones(n)

        # One final scale keeps the usual case exactly mean-one; the bisection
        # above keeps the clipped bounds valid even for high-contrast vectors.
        arr = np.clip(arr / mean, MIN_PRECISION, MAX_PRECISION)
        return arr.astype(float, copy=False)
    except Exception:
        return _ones(n)


def robust_scale(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Robustly center and scale a vector using median absolute deviation."""
    arr = np.asarray(values, dtype=float)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median))
    return (arr - median) / (mad + eps)
