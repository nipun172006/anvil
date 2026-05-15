"""Official-submission adapter with selectable precision-agent modes."""

from __future__ import annotations

import os

import numpy as np

from adapter import Adapter

from src.agents.adaptive_topk_spread import AdaptiveTopKSpreadAgent
from src.agents.geometry import GeometryPrecisionAgent
from src.agents.hessian_diag_geometry import HessianDiagonalGeometryAgent
from src.agents.hessian_ruiz_geometry import HessianRuizGeometryAgent
from src.agents.hybrid import HybridPrecisionAgent
from src.agents.identity import IdentityPrecisionAgent
from src.agents.magnitude import MagnitudePrecisionAgent
from src.agents.nearest_pattern import NearestPatternPrecisionAgent
from src.agents.spread_optimized_geometry import SpreadOptimizedGeometryAgent
from src.agents.topk_consensus import TopKConsensusPrecisionAgent
from src.agents.topk_geom import TopKGeometryPrecisionAgent
from src.agents.topk_hessian import (
    TopKHessianDiagonalPrecisionAgent,
    TopKHessianRuizPrecisionAgent,
)
from src.utils.normalization import sanitize_precision


AGENT_CLASSES = {
    "adaptive_topk_spread": AdaptiveTopKSpreadAgent,
    "identity": IdentityPrecisionAgent,
    "magnitude": MagnitudePrecisionAgent,
    "nearest": NearestPatternPrecisionAgent,
    "spread_opt": SpreadOptimizedGeometryAgent,
    "topk": TopKConsensusPrecisionAgent,
    "topk_geom": TopKGeometryPrecisionAgent,
    "topk_hdiag": TopKHessianDiagonalPrecisionAgent,
    "topk_ruiz": TopKHessianRuizPrecisionAgent,
    "geometry": GeometryPrecisionAgent,
    "hessian_diag": HessianDiagonalGeometryAgent,
    "hessian_ruiz": HessianRuizGeometryAgent,
    "hybrid": HybridPrecisionAgent,
}
DEFAULT_MODE = "adaptive_topk_spread"


class Engine(Adapter):
    """Adapter-compatible precision controller.

    The official harness instantiates this class with stored patterns and frozen
    PCAM parameters, then calls ``predict_precision`` once per corrupted query.
    """

    def __init__(self, stored_patterns: np.ndarray, model_params: dict | None = None):
        self.X = np.asarray(stored_patterns, dtype=float)
        self.N = int(self.X.shape[1])
        self.model_params = model_params or {}
        self.mode = os.environ.get("AGENT_MODE", DEFAULT_MODE).strip().lower()
        agent_cls = AGENT_CLASSES.get(self.mode, HybridPrecisionAgent)
        if self.mode not in AGENT_CLASSES:
            self.mode = DEFAULT_MODE
        try:
            self.agent = agent_cls(self.X, self.model_params)
        except Exception:
            self.agent = None

    def predict_precision(self, corrupted_query: np.ndarray) -> np.ndarray:
        """Return a positive, finite, mean-one precision vector of shape ``(N,)``."""
        try:
            if self.agent is None:
                return np.ones(self.N, dtype=float)
            precision = self.agent.predict_precision(corrupted_query)
            return sanitize_precision(precision, self.N)
        except Exception:
            return np.ones(self.N, dtype=float)
