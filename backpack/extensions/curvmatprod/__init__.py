"""
Curvature-matrix product backPACK extensions.

Those extension propagate additional information through the computation graph.
They are more expensive to run than a standard gradient backpropagation.

This extension does not compute information directly, but gives access to
functions to compute Matrix-Matrix products with Block-Diagonal approximations
of the curvature, such as the Block-diagonal Generalized Gauss-Newton
"""

from .ggnmp import GGNMP
from .hmp import HMP
from .pchmp import PCHMP

__all__ = [
    "GGNMP",
    "HMP",
    "PCHMP",
]
