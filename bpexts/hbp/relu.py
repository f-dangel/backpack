"""Elementwise ReLU activation function with HBP functionality."""

from .nonlinear import hbp_elementwise_nonlinear
from torch import (gt, zeros_like)
from torch.nn import ReLU


class HBPReLU(hbp_elementwise_nonlinear(ReLU)):
    """Componentwise ReLU layer with HBP functionality.

    Applies ReLU(x) = max(x, 0) elementwise.
    """
    # override
    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, ReLU):
            raise ValueError("Expecting torch.nn.ReLU, got {}".format(
                torch_layer.__class__))
        # create instance
        relu = cls()
        return relu

    # override
    def hbp_derivative_hooks(self):
        """Hooks for access to first and second derivative."""
        self.register_exts_forward_hook(self.store_relu_derivatives)

    @staticmethod
    def store_relu_derivatives(module, input, output):
        """Compute relu'(input) and relu''(input).

        Intended use as forward-hook.
        Initialize buffers 'grad_phi' and 'gradgrad_phi'.
        """
        if not len(input) == 1:
            raise ValueError('Cannot handle multi-input scenario')
        # .to() statement casts to same type
        grad_relu = gt(input[0], 0).to(input[0]).detach()
        gradgrad_relu = zeros_like(input[0]).detach()
        module.register_exts_buffer('grad_phi', grad_relu)
        module.register_exts_buffer('gradgrad_phi', gradgrad_relu)
