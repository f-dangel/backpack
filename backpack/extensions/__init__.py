"""BackPACK extensions that can be passed into a ``with backpack(...)`` context."""

from .curvmatprod import GGNMP, HMP, PCHMP
from .firstorder import BatchGrad, BatchL2Grad, SumGradSquared, Variance
from .secondorder import (
    HBP,
    KFAC,
    KFLR,
    KFRA,
    BatchDiagGGNExact,
    BatchDiagGGNMC,
    BatchDiagHessian,
    DiagGGNExact,
    DiagGGNMC,
    DiagHessian,
    SqrtGGNExact,
    SqrtGGNMC,
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
    "BatchDiagGGNMC",
    "DiagHessian",
    "BatchDiagHessian",
    "SqrtGGNExact",
    "SqrtGGNMC",
]
