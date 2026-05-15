"""Direct spread objective utilities for precision-weighted Hessians."""

from __future__ import annotations

import os

import numpy as np


PI_MIN = 0.1
PI_MAX = 10.0
TINY = 1e-9


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


def project_precision(pi: np.ndarray, n: int) -> np.ndarray:
    """Clip precision to official bounds and mean-normalize."""
    arr = np.asarray(pi, dtype=float).reshape(-1)
    if arr.shape != (int(n),) or not np.all(np.isfinite(arr)):
        return np.ones(int(n), dtype=float)
    arr = np.nan_to_num(arr, nan=1.0, posinf=PI_MAX, neginf=PI_MIN)
    arr = np.clip(arr, PI_MIN, PI_MAX)
    mean = float(np.mean(arr))
    if mean <= 0.0 or not np.isfinite(mean):
        return np.ones(int(n), dtype=float)
    arr = arr / mean
    arr = np.clip(arr, PI_MIN, PI_MAX)
    mean = float(np.mean(arr))
    if mean > 0.0 and np.isfinite(mean):
        arr = arr / mean
    return arr


def stable_condition_number(H: np.ndarray, pi: np.ndarray) -> float:
    """Condition number of diag(sqrt(pi)) @ H @ diag(sqrt(pi)).

    Mirrors the official spread computation for stable basins: eigenvalues are
    taken with ``eigvalsh`` after symmetrization, and non-positive minima are
    treated as a large penalty rather than optimized toward.
    """
    matrix = np.asarray(H, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return float("inf")
    n = matrix.shape[0]
    precision = project_precision(pi, n)
    try:
        pi_sqrt = np.sqrt(np.maximum(precision, TINY))
        S = (pi_sqrt[:, None] * matrix) * pi_sqrt[None, :]
        S = 0.5 * (S + S.T)
        eigs = np.linalg.eigvalsh(S)
    except Exception:
        return float("inf")

    if eigs.size < 2 or not np.all(np.isfinite(eigs)):
        return float("inf")
    max_eig = float(np.max(eigs))
    min_eig = float(np.min(eigs))
    if min_eig <= TINY:
        return float("inf")
    return max_eig / min_eig


def _pi_from_log(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    y = y - float(np.max(y))
    pi = np.exp(y)
    return project_precision(pi, y.size)


def _log_from_pi(pi: np.ndarray) -> np.ndarray:
    projected = project_precision(pi, np.asarray(pi).size)
    return np.log(np.maximum(projected, TINY))


def _axis_direction(H: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Heuristic direction from smallest/largeest modes of current S."""
    n = H.shape[0]
    precision = project_precision(pi, n)
    pi_sqrt = np.sqrt(np.maximum(precision, TINY))
    S = (pi_sqrt[:, None] * H) * pi_sqrt[None, :]
    S = 0.5 * (S + S.T)
    eigvals, eigvecs = np.linalg.eigh(S)
    if eigvals[0] <= TINY or not np.all(np.isfinite(eigvals)):
        return np.zeros(n, dtype=float)
    low = eigvecs[:, 0] ** 2
    high = eigvecs[:, -1] ** 2
    direction = low - high
    direction = direction - float(np.mean(direction))
    norm = float(np.linalg.norm(direction))
    if norm <= TINY or not np.isfinite(norm):
        return np.zeros(n, dtype=float)
    return direction / norm


def optimize_precision_for_hessian(
    H: np.ndarray,
    n: int,
    seed: int = 0,
    budget: int = 100,
) -> np.ndarray:
    """Minimize Hessian spread with a small log-space stochastic search."""
    matrix = np.asarray(H, dtype=float)
    n = int(n)
    if matrix.shape != (n, n):
        return np.ones(n, dtype=float)

    budget = _env_int("SPREAD_OPT_BUDGET", int(budget))
    scale0 = max(_env_float("SPREAD_OPT_SCALE", 0.4), 1e-6)
    seed = _env_int("SPREAD_OPT_SEED", int(seed))
    rng = np.random.default_rng(seed)

    best_pi = np.ones(n, dtype=float)
    best_score = stable_condition_number(matrix, best_pi)
    if not np.isfinite(best_score):
        return best_pi

    best_y = _log_from_pi(best_pi)
    current_y = best_y.copy()
    current_pi = best_pi.copy()
    current_score = best_score

    for t in range(max(0, budget)):
        frac = t / max(1, budget - 1)
        scale = scale0 * (0.15 ** frac)

        if t % 4 == 0:
            direction = _axis_direction(matrix, current_pi)
            if not np.any(direction):
                direction = rng.standard_normal(n)
        elif t % 4 == 1:
            direction = _axis_direction(matrix, best_pi)
            if not np.any(direction):
                direction = rng.standard_normal(n)
        elif t % 4 == 2:
            direction = rng.standard_normal(n)
        else:
            direction = np.zeros(n, dtype=float)
            idx = int(rng.integers(0, n))
            direction[idx] = rng.choice(np.array([-1.0, 1.0]))

        direction = direction - float(np.mean(direction))
        norm = float(np.linalg.norm(direction))
        if norm <= TINY or not np.isfinite(norm):
            continue
        direction = direction / norm

        accepted = False
        for signed_scale in (scale, -scale):
            trial_y = current_y + signed_scale * direction
            trial_pi = _pi_from_log(trial_y)
            score = stable_condition_number(matrix, trial_pi)
            if score < current_score:
                current_y = _log_from_pi(trial_pi)
                current_pi = trial_pi
                current_score = score
                accepted = True
                if score < best_score:
                    best_score = score
                    best_pi = trial_pi
                    best_y = current_y.copy()
                break

        if not accepted and t % 8 == 7:
            current_y = best_y.copy()
            current_pi = best_pi.copy()
            current_score = best_score

    return project_precision(best_pi, n)
