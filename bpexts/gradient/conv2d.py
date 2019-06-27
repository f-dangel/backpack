"""Extension of torch.nn.Conv2d for computing batch gradients."""

import torch.nn
from torch import einsum
from . import config
from .config import CTX


class Conv2d(torch.nn.Conv2d):
    """Extended backpropagation for torch.nn.Conv2d."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unfold = torch.nn.Unfold(
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=self.padding,
            stride=self.stride)
        self.register_forward_pre_hook(self.store_input)
        self.register_backward_hook(self.compute_first_order_info)

    @staticmethod
    def store_input(module, input):
        """Pre forward hook saving layer input as buffer.

        Initialize module buffer `ìnput`.
        """
        if not len(input) == 1:
            raise ValueError('Cannot handle multi-input scenario')
        if not len(input[0].size()) == 4:
            raise ValueError('Expecting 4D input (batch, channel, x, y)')
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
            # assume the matrix has three axes (batch, output, classes)
            assert len(shape) == 3
            assert shape[0] == module.input.size(0)
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
            self.bias.diag_ggn = einsum('bic->i', (sqrt_ggn_bias**2))
        raise NotImplementedError
        if self.weight.requires_grad:
            # TODO: Combine into a single (more memory-efficient) einsum
            self.weight.diag_ggn = einsum('bic,bj->ij',
                                          (sqrt_ggn_out**2, self.input**2))

    def _update_backpropagated_sqrt_ggn(self, sqrt_ggn_out):
        """Apply transposed Jacobian of module output with respect to input.

        Update ``CTX._backpropagated_sqrt_ggn``.
        """
        # TODO need access to dimensions of input AND output
        raise NotImplementedError
        CTX._backpropagated_sqrt_ggn = einsum('ij,bic->bjc',
                                              (self.weight, sqrt_ggn_out))

    def compute_grad_batch(self, grad_output):
        """Compute individual gradients for module parameters.

        Store bias batch gradients in `module.bias.batch_grad` and
        weight batch gradients in `module.weight.batch_grad`.
        """
        if self.bias is not None and self.bias.requires_grad:
            self.bias.grad_batch = self._compute_bias_grad_batch(grad_output)
        if self.weight.requires_grad:
            self.weight.grad_batch = self._compute_weight_grad_batch(
                grad_output)

    def _compute_bias_grad_batch(self, grad_output):
        """Compute bias batch gradients.

        The batchwise gradient of a linear layer is simply given
        by the gradient with respect to the layer's output, summed over
        the spatial dimensions (each number in the bias vector is added.
        to the spatial output of an entire channel).
        """
        return grad_output.sum(3).sum(2)

    def _compute_weight_grad_batch(self, grad_output):
        """Compute weight batch gradients.

        The linear layer applies
        Y = W * X
        (neglecting the bias) to a single sample X, where W denotes the
        matrix view of the kernel, Y is a view of the output and X denotes
        the unfolded input matrix.

        Note on shapes/dims:
        --------------------
        original input x: (batch_size, in_channels, x_dim, y_dim)
        original kernel w: (out_channels, in_channels, k_x, k_y)
        im2col input X: (batch_size, num_patches, in_channels * k_x * k_y)
        kernel matrix W: (out_channels, in_channels *  k_x * k_y)
        matmul result Y: (batch_size, out_channels, num_patches)
                       = (batch_size, out_channels, x_out_dim * y_out_dim)
        col2im output y: (batch_size, out_channels, x_out_dim, y_out_dim)

        Forward pass: (pseudo)
        -------------
        X = unfold(x) = im2col(x)
        W = view(w)
        Y[b,i,j] = W[i,m] *  X[b,m,j]
        y = view(Y)

        Backward pass: (pseudo)
        --------------
        Given: dE/dy    (same shape as y)
        dE/dY = view(dE/dy) (same shape as Y)

        dE/dW[b,k,l]    (batch-wise gradient)
        dE/dW[b,k,l] = (dY[b,i,j]/dW[k,l]) * dE/dY[b,i,j]
                     = delta(i,k) * delta(m,l) * X[b,m,j] * dE/dY[b,i,j]
                     = delta(m,l) * X[b,m,j] * dE/dY[b,k,j]

        Result:
        -------
        dE/dw = view(dE/dW)
        """
        batch_size = grad_output.size(0)
        dE_dw_shape = (batch_size, ) + self.weight.size()
        X = self.unfold(self.input)
        dE_dY = grad_output.view(batch_size, self.out_channels, -1)
        dE_dW = einsum('bml,bkl->bmk', (dE_dY, X))
        return dE_dW.view(dE_dw_shape)

    def compute_sum_grad_squared(self, grad_output):
        """Square the gradients for each sample and sum over the batch."""
        if self.bias is not None and self.bias.requires_grad:
            self.bias.sum_grad_squared = self._compute_bias_sgs(grad_output)
        if self.weight.requires_grad:
            self.weight.sum_grad_squared = self._compute_weight_sgs(
                grad_output)

    def _compute_weight_sgs(self, grad_output):
        return (self._compute_weight_grad_batch(grad_output)**2).sum(0)

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
