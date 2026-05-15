"""Precision-agent implementations."""

from src.agents.geometry import GeometryPrecisionAgent
from src.agents.adaptive_topk_spread import AdaptiveTopKSpreadAgent
from src.agents.hessian_diag_geometry import HessianDiagonalGeometryAgent
from src.agents.hessian_ruiz_geometry import HessianRuizGeometryAgent
from src.agents.hybrid import HybridPrecisionAgent
from src.agents.identity import IdentityPrecisionAgent
from src.agents.magnitude import MagnitudePrecisionAgent
from src.agents.nearest_pattern import NearestPatternPrecisionAgent
from src.agents.spread_optimized_geometry import SpreadOptimizedGeometryAgent
from src.agents.topk_geom import TopKGeometryPrecisionAgent
from src.agents.topk_hessian import (
    TopKHessianDiagonalPrecisionAgent,
    TopKHessianRuizPrecisionAgent,
)
from src.agents.topk_consensus import TopKConsensusPrecisionAgent

__all__ = [
    "AdaptiveTopKSpreadAgent",
    "GeometryPrecisionAgent",
    "HessianDiagonalGeometryAgent",
    "HessianRuizGeometryAgent",
    "HybridPrecisionAgent",
    "IdentityPrecisionAgent",
    "MagnitudePrecisionAgent",
    "NearestPatternPrecisionAgent",
    "SpreadOptimizedGeometryAgent",
    "TopKGeometryPrecisionAgent",
    "TopKHessianDiagonalPrecisionAgent",
    "TopKHessianRuizPrecisionAgent",
    "TopKConsensusPrecisionAgent",
]
