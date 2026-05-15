"""Hessian helpers matching the official PCAM model formula."""

from __future__ import annotations

import numpy as np


def stable_softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for a one-dimensional score vector."""
    scores = np.asarray(z, dtype=float).reshape(-1)
    if scores.size == 0:
        return np.ones(0, dtype=float)
    scores = scores - float(np.max(scores))
    exp_scores = np.exp(scores)
    total = float(np.sum(exp_scores))
    if total <= 0.0 or not np.isfinite(total):
        return np.ones_like(scores, dtype=float) / scores.size
    return exp_scores / total


def compute_hessian(
    a: np.ndarray,
    X: np.ndarray,
    R: np.ndarray,
    eta: float,
    beta: float,
    jitter: float = 0.0,
) -> np.ndarray:
    """Compute the official PCAM Hessian at state ``a``.

    H(a) = R - eta * beta * X.T @ (diag(s) - outer(s, s)) @ X
    where s = softmax(beta * X @ a).
    """
    patterns = np.asarray(X, dtype=float)
    state = np.asarray(a, dtype=float).reshape(-1)
    operator = np.asarray(R, dtype=float)
    if patterns.ndim != 2:
        raise ValueError("X must be a 2D array")
    if state.shape != (patterns.shape[1],):
        raise ValueError("a must have shape (N,)")
    if operator.shape != (patterns.shape[1], patterns.shape[1]):
        raise ValueError("R must have shape (N, N)")

    eta = float(eta)
    beta = float(beta)
    s = stable_softmax(beta * (patterns @ state))
    C = np.diag(s) - np.outer(s, s)
    H = operator - eta * beta * (patterns.T @ (C @ patterns))
    H = 0.5 * (H + H.T)
    if jitter > 0.0:
        H = H + float(jitter) * np.eye(patterns.shape[1])
    return H
