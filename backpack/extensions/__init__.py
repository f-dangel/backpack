"""
BackPACK Extensions
"""

from .curvmatprod import CMP, GGNMP, HMP, PCHMP
from .firstorder import BatchGrad, BatchL2Grad, SumGradSquared, Variance
from .secondorder import (
    KFAC,
    KFLR,
    KFRA,
    HBP,
    DiagGGNExact,
    DiagGGNMC,
    DiagGGN,
    DiagHessian,
)

__all__ = [
    "PCHMP",
    "GGNMP",
    "HMP",
    "CMP",
    "BatchL2Grad",
    "BatchGrad",
    "SumGradSquared",
    "Variance",
    "KFAC",
    "KFLR",
    "KFRA",
    "HBP",
    "DiagGGNExact",
    "DiagGGNMC",
    "DiagGGN",
    "DiagHessian",
]
