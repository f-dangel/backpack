"""Block-diagonal curvature products
====================================

These extensions do not compute information directly, but give access to
functions to compute matrix-matrix products with block-diagonal approximations
of the Hessian.

Extensions propagate functions through the computation graph. In contrast to
standard gradient computation, the graph is retained during backpropagation
(this results in higher memory consumption). The cost of one matrix-vector
multiplication is on the order of one backward pass.

Implemented extensions are matrix-free curvature-matrix multiplication with
the block-diagonal of the Hessian, generalized Gauss-Newton (GGN)/Fisher, and
positive-curvature Hessian. They are formalized by the concept of Hessian
backpropagation, described in:

- `Modular Block-diagonal Curvature Approximations for Feedforward Architectures
  <http://proceedings.mlr.press/v108/dangel20a/dangel20a.pdf>`_
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
