"""Second order extensions.
====================================

Second-order extensions propagate additional information through the graph
to extract structural or local approximations to second-order information.
They are more expensive to run than a standard gradient backpropagation.
The implemented extensions are

- The diagonal of the Generalized Gauss-Newton (GGN)/Fisher information,
  using exact computation
  (:func:`DiagGGNExact <backpack.extensions.DiagGGNExact>`)
  or Monte-Carlo approximation
  (:func:`DiagGGNMC <backpack.extensions.DiagGGNMC>`).
- Kronecker Block-Diagonal approximations of the GGN/Fisher
  :func:`KFAC <backpack.extensions.KFAC>`,
  :func:`KFRA <backpack.extensions.KFRA>`,
  :func:`KFLR <backpack.extensions.KFLR>`.
- The diagonal of the Hessian :func:`DiagHessian <backpack.extensions.DiagHessian>`
"""

from .diag_ggn import DiagGGN, DiagGGNExact, DiagGGNMC
from .diag_hessian import DiagHessian
from .hbp import HBP, KFAC, KFLR, KFRA

__all__ = [
    "DiagGGNExact",
    "DiagGGNMC",
    "DiagGGN",
    "DiagHessian",
    "KFAC",
    "KFLR",
    "KFRA",
    "HBP",
]
