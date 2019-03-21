"""Hessian backpropagation for the reshape operation."""

from torch.nn import Module
from .module import hbp_decorate


class HBPReshape(hbp_decorate(Module)):
    """Reshape operation with Hessian backpropagation functionality.

    Parameters:
    -----------
    shape : tuple(int)
        Tuple of integers describing the target shape. Same conventions
        as ``torch.reshape``.

    Details:
    --------
    The HBP procedure for a reshape of the quantity is trivial, since the
    Hessian with respect to the vectorized quantity is the same. So this operation
    simply passes on the Hessian that it is provided.
    """

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
