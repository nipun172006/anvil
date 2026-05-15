"""Shared utility functions for precision agents."""

from src.utils.distances import cosine_distances, euclidean_distances, safe_l2_normalize
from src.utils.hessian import compute_hessian, stable_softmax
from src.utils.normalization import sanitize_precision
from src.utils.spread_objective import (
    optimize_precision_for_hessian,
    stable_condition_number,
)

__all__ = [
    "compute_hessian",
    "cosine_distances",
    "euclidean_distances",
    "optimize_precision_for_hessian",
    "safe_l2_normalize",
    "sanitize_precision",
    "stable_condition_number",
    "stable_softmax",
]
