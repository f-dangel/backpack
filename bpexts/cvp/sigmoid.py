"""Elementwise sigmoid activation function with CVP functionality."""

from .nonlinear import cvp_elementwise_nonlinear
from torch.nn import Sigmoid


class CVPSigmoid(cvp_elementwise_nonlinear(Sigmoid)):
    """Componentwise sigmoid layer with CVP functionality.

    Applies sigma(x) = 1 / (1 + exp(-x)) elementwise.
    """
    # override
    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, Sigmoid):
            raise ValueError("Expecting torch.nn.Sigmoid, got {}".format(
                torch_layer.__class__))
        # create instance
        sigmoid = cls()
        return sigmoid

    # override
    def cvp_derivative_hooks(self):
        """Hooks for access to first and second derivative."""
        self.register_exts_forward_hook(self.store_sigmoid_derivatives)

    @staticmethod
    def store_sigmoid_derivatives(module, input, output):
        """Compute sigma'(input) and sigma''(input).

        Use recursiveness sigma' = sigma * (1 - sigma) to avoid
        unnecessary reevaluation of the sigmoid function.

        Use recursiveness sigma'' = sigma' * (1 - 2 * sigma) to
        avoid redundant evaluations of the sigmoid function.

        Intended use as forward-hook.
        Initialize buffers 'grad_phi' and 'gradgrad_phi'.
        """
        sigma = output
        grad_sigma = (sigma * (1 - sigma)).detach()
        gradgrad_sigma = (grad_sigma * (1 - 2 * sigma)).detach()
        module.register_exts_buffer('grad_phi', grad_sigma)
        module.register_exts_buffer('gradgrad_phi', gradgrad_sigma)
