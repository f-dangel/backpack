"""Extension of torch.nn.ReLU for computing first-order information."""

import torch.nn
from torch import einsum
from . import config
from .config import CTX


class ReLU(torch.nn.ReLU):
    """Extended gradient backpropagation for torch.nn.ReLU."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_forward_pre_hook(self.store_input)
        self.register_backward_hook(self.compute_first_order_info)

    @staticmethod
    def store_input(module, input):
        """Pre forward hook saving layer input as buffer.

        Initialize module buffer `ìnput`.
        """
        if not len(input) == 1:
            raise ValueError('Cannot handle multi-input scenario')
        if not len(input[0].size()) == 2:
            raise ValueError('Expecting 2D input (batch, data)')
        module.register_buffer('input', input[0].clone().detach())

    @staticmethod
    def compute_first_order_info(module, grad_input, grad_output):
        """Check which quantities need to be computed and evaluate them."""
        if CTX.is_active(config.DIAG_GGN):
            sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
            # some sanity checks
            shape = tuple(sqrt_ggn_out.size())
            assert len(shape) == 3
            assert shape[0] == module.input.size(0)
            assert shape[1] == module.input.size(1)
            # update the backpropagated quantity by application of the Jacobian
            module._update_backpropagated_sqrt_ggn(sqrt_ggn_out)

    def _update_backpropagated_sqrt_ggn(self, sqrt_ggn_out):
        """Apply transposed Jacobian of module output with respect to input.

        Update ``CTX._backpropagated_sqrt_ggn``.
        """
        d_relu = torch.gt(self.input, 0)
        CTX._backpropagated_sqrt_ggn = einsum('bi,bic->bic',
                                              (d_relu, sqrt_ggn_out))
