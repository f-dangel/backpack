"""
Second order backPACK extensions.

Those extension propagate additional information through the computation graph.
They are more expensive to run than a standard gradient backpropagation.

Those extension make it easier to extract structural or local approximations
to second-order information, such as
- `DiagHessian`: The diagonal of the Hessian.
- `DiagGGN`: The diagonal of the Generalized Gauss-Newton
  (or Fisher information matrix), supports exact computation or sampling.
  - `DiagGGNExact`: Exact diagonal of the GGN
  - `DiagGGNMC`: MC-sampled diagonal of the GGN/Fisher
- `KFAC`, `KFRA`, `KFLR`: Kronecker Block-Diagonal approximations of the
  Generalized Gauss-Newton (or Fisher information matrix).
- `HBP`: A general framework that encompasses KFAC, KFRA and KFLR.
"""

from .diag_ggn import DiagGGN, DiagGGNExact, DiagGGNMC
from .diag_hessian import DiagHessian
from .hbp import HBP, KFAC, KFLR, KFRA
