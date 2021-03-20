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

<<<<<<< HEAD
from .diag_ggn import BatchDiagGGNExact, DiagGGNExact, DiagGGNMC
from .diag_hessian import BatchDiagHessian, DiagHessian

=======
from .diag_ggn import BatchDiagGGNExact, DiagGGNExact, DiagGGNMC
from .diag_hessian import DiagHessian

>>>>>>> 6e2f6ace71d1aac118f878f968753ac9e83f742d
from .hbp import HBP, KFAC, KFLR, KFRA

__all__ = [
    "DiagGGNExact",
    "BatchDiagGGNExact",
    "DiagGGNMC",
    "DiagHessian",
    "BatchDiagHessian",
    "KFAC",
    "KFLR",
    "KFRA",
    "HBP",
]
