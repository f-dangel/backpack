"""Elementwise sigmoid activation function with HBP functionality."""

from .nonlinear import hbp_elementwise_nonlinear
from torch.nn import Sigmoid


class HBPSigmoid(hbp_elementwise_nonlinear(Sigmoid)):
    """Componentwise sigmoid layer with HBP functionality.

    Applies sigma(x) = 1 / (1 + exp(-x)) elementwise.
    """
    # override
    def hbp_derivative_hooks(self):
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
