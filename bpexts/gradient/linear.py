"""Extension of torch.nn.Linear for computing first-order information."""

import torch.nn
from torch import einsum
from . import config
from .config import CTX


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

        if CTX.is_active(config.BATCH_GRAD):
            module.compute_grad_batch(grad_out)
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

    def compute_grad_batch(self, grad_output):
        """Compute batchwise gradients of module parameters.

        Store bias batch gradients in `module.bias.batch_grad` and
        weight batch gradients in `module.weight.batch_grad`.
        """
        if self.bias is not None and self.bias.requires_grad:
            self.bias.grad_batch = self._compute_bias_grad_batch(grad_output)
        if self.weight.requires_grad:
            self.weight.grad_batch = self._compute_weight_grad_batch(
                grad_output)

    def _compute_bias_grad_batch(self, grad_output):
        """Compute bias batch gradients from grad w.r.t. layer outputs.

        The batchwise gradient of a linear layer is simply given
        by the gradient with respect to the layer's output.
        """
        return grad_output

    def _compute_weight_grad_batch(self, grad_output):
        r"""Compute weight batch gradients from grad w.r.t layer outputs.

        The linear layer applies
        y = W x + b.
        to a single sample x, where W denotes the weight and b the bias.

        Result:
        -------
        Finally, this yields
        dE / d vec(W) = (dy / d vec(W))^T dy  (vec denotes row stacking)
                      = dy \otimes x^T

        Derivation:
        -----------
        The Jacobian for W is given by
        (matrix derivative notation)
        dy / d vec(W) = x^T \otimes I    (vec denotes column stacking),
        dy / d vec(W) = I \otimes x      (vec denotes row stacking),
        (dy / d vec(W))^T = I \otimes x^T  (vec denotes row stacking)
        or
        (index notation)
        dy[i] / dW[j,k] = delta(i,j) x[k]    (index notation).
        """
        batch_size = grad_output.size(0)
        weight_grad_batch = einsum('bi,bj->bij', (grad_output, self.input))
        return weight_grad_batch.view(batch_size, self.out_features,
                                      self.in_features)

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
        return (self._compute_bias_grad_batch(grad_output)**2).sum(0)

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
