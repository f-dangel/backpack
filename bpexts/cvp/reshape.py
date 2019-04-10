"""Curvature-vector products for reshape operation."""

from torch.nn import Module
from ..hbp.module import hbp_decorate


class CVPReshape(hbp_decorate(Module)):
    """Reshape layer with recursive Hessian-vector products."""

    def __init__(self, shape):
        super().__init__()
        self._target_shape = shape

    def forward(self, input):
        """Apply the transposition operation."""
        return input.reshape(*self._target_shape)

    # override
    def hbp_hooks(self):
        """No hooks required."""
        pass

    # override
    def input_hessian(self, output_hessian, modify_2nd_order_terms='none'):
        """Pass on the Hessian with respect to the layer input."""
        return output_hessian
