"""Extension of torch.nn.Linear for computing first-order information."""

import torch.nn
from torch import einsum
from . import config
from .config import CTX
from .batchgrad.linear import bias_grad_batch


class Linear(torch.nn.Linear):
    """Extended gradient backpropagation for torch.nn.Linear."""

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
        if not len(grad_output) == 1:
            raise ValueError('Cannot handle multi-output scenario')

        grad_out = grad_output[0].clone().detach()

        if CTX.is_active(config.SUM_GRAD_SQUARED):
            module.compute_sum_grad_squared(grad_out)
        if CTX.is_active(config.DIAG_GGN):
            sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
            # some sanity checks
            shape = tuple(sqrt_ggn_out.size())
            assert len(shape) == 3
            assert shape[0] == module.input.size(0)
            assert shape[1] == module.out_features
            # compute the diagonal of the GGN
            module._extract_diag_ggn(sqrt_ggn_out)
            # update the backpropagated quantity by application of the Jacobian
            module._update_backpropagated_sqrt_ggn(sqrt_ggn_out)

    def _extract_diag_ggn(self, sqrt_ggn_out):
        """Obtain sqrt representation of GGN. Extract diagonal.

        Initialize ``weight.diag_ggn`` and ``bias.diag_ggn``.
        """
        if self.bias is not None and self.bias.requires_grad:
            sqrt_ggn_bias = sqrt_ggn_out
            self.bias.diag_ggn = einsum('bic,bic->i',
                                        (sqrt_ggn_bias, sqrt_ggn_bias))
        if self.weight.requires_grad:
            # TODO: Combine into a single (more memory-efficient) einsum
            sqrt_ggn_weight = einsum('bic,bj->bijc',
                                     (sqrt_ggn_out, self.input))
            self.weight.diag_ggn = einsum('bijc,bijc->ij',
                                          (sqrt_ggn_weight, sqrt_ggn_weight))

    def _update_backpropagated_sqrt_ggn(self, sqrt_ggn_out):
        """Apply transposed Jacobian of module output with respect to input.

        Update ``CTX._backpropagated_sqrt_ggn``.
        """
        CTX._backpropagated_sqrt_ggn = einsum('ij,bic->bjc',
                                              (self.weight, sqrt_ggn_out))

    def compute_sum_grad_squared(self, grad_output):
        """Square the gradients for each sample and sum over the batch."""
        if self.bias is not None and self.bias.requires_grad:
            self.bias.sum_grad_squared = self._compute_bias_sgs(grad_output)
        if self.weight.requires_grad:
            self.weight.sum_grad_squared = self._compute_weight_sgs(
                grad_output)

    def _compute_weight_sgs(self, grad_output):
        return einsum('bi,bj->ij', (grad_output**2, self.input**2))

    def _compute_bias_sgs(self, grad_output):
        return (bias_grad_batch(self, grad_output)**2).sum(0)

    def clear_grad_batch(self):
        if hasattr(self.weight, "grad_batch"):
            del self.weight.grad_batch
        if hasattr(self.bias, "grad_batch"):
            del self.bias.grad_batch

    def clear_sum_grad_squared(self):
        if hasattr(self.weight, "sum_grad_squared"):
            del self.weight.sum_grad_squared
        if hasattr(self.bias, "sum_grad_squared"):
            del self.bias.sum_grad_squared

    def clear_diag_ggn(self):
        if hasattr(self.weight, "diag_ggn"):
            del self.weight.diag_ggn
        if hasattr(self.bias, "diag_ggn"):
            del self.bias.diag_ggn
