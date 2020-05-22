"""Block-diagonal curvature-matrix products (function backpropagation).
=======================================================================

These extensions do not compute information directly, but give access to
functions to compute matrix-matrix products with block-diagonal approximations
of the curvature, such as the block-diagonal generalized Gauss-Newton or Hessian.

Extensions propagate functions through the computation graph. In contrast to
standard gradient computation, the graph is retained during backpropagation
(this results in higher memory consumption). The cost of one matrix-vector
multiplication is on the order of one backward pass.

Implemented extensions are matrix-free curvature-matrix multiplication with

- The block-diagonal generalized Gauss-Newton (GGN)/Fisher information
  (:func:`GGNMP <backpack.extensions.GGNMP>`).
- The block-diagonal Hessian (:func:`HMP <backpack.extensions.HMP>`).
- The block-diagonal positive-curvature Hessian (PCH,
  :func:`PCHMP <backpack.extensions.PCHMP>`) introduced in
  - `BDA-PCH: Block-Diagonal Approximation of Positive-Curvature Hessian for
    Training Neural Networks <https://arxiv.org/abs/1802.06502v2>`_
    by Sheng-Wei Chen, Chun-Nan Chou and Edward Y. Chang, 2018.

The following paper describes the details of Hessian backpropagation:

- `Modular Block-diagonal Curvature Approximations for Feedforward Architectures
  <https://arxiv.org/abs/1802.06502v2>`_
  by Felix Dangel, Stefan Harmeling, Philipp Hennig, 2020.
"""

from .ggnmp import GGNMP
from .hmp import HMP
from .pchmp import PCHMP

__all__ = [
    "GGNMP",
    "HMP",
    "PCHMP",
]
