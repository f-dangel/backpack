"""
BackPACK Extensions
"""

from .curvmatprod import GGNMP, HMP, PCHMP
from .firstorder import BatchGrad, BatchL2Grad, SumGradSquared, Variance
from .secondorder import (
    HBP,
    KFAC,
    KFLR,
    KFRA,
    DiagGGN,
    DiagGGNExact,
    BatchDiagGGNExact,
    DiagGGNMC,
    DiagHessian,
    BatchDiagHessian,
)

__all__ = [
    "PCHMP",
    "GGNMP",
    "HMP",
    "BatchL2Grad",
    "BatchGrad",
    "SumGradSquared",
    "Variance",
    "KFAC",
    "KFLR",
    "KFRA",
    "HBP",
    "DiagGGNExact",
    "BatchDiagGGNExact",
    "DiagGGNMC",
    "DiagGGN",
    "DiagHessian",
    "BatchDiagHessian",
]
