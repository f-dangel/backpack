"""Elementwise Tanh activation function with CVP functionality."""

from .nonlinear import cvp_elementwise_nonlinear
from torch.nn import Tanh


class CVPTanh(cvp_elementwise_nonlinear(Tanh)):
    """Componentwise tanh layer with CVP functionality.

    Applies tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x)) elementwise.
    """
    # override
    @classmethod
    def from_torch(cls, torch_layer):
        if not isinstance(torch_layer, Tanh):
            raise ValueError("Expecting torch.nn.Tanh, got {}".format(
                torch_layer.__class__))
        # create instance
        tanh = cls()
        return tanh

    # override
    def cvp_derivative_hooks(self):
        """Hooks for access to first and second derivative."""
        self.register_exts_forward_hook(self.store_tanh_derivatives)

    @staticmethod
    def store_tanh_derivatives(module, input, output):
        """Compute tanh'(input) and tanh''(input).

        Use recursiveness tanh' = 1 - tanh**2 to avoid
        unnecessary reevaluation of the tanh function.

        Use recursiveness tanh'' = -2 * tanh * tanh' to
        avoid redundant evaluations of the tanh function.

        Intended use as forward-hook.
        Initialize buffers 'grad_phi' and 'gradgrad_phi'.
        """
        tanh = output
        grad_tanh = (1. - tanh**2).detach()
        gradgrad_tanh = (-2. * tanh * grad_tanh).detach()
        module.register_exts_buffer('grad_phi', grad_tanh)
        module.register_exts_buffer('gradgrad_phi', gradgrad_tanh)
