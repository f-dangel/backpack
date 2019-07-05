import torch.nn
from ....utils import einsum
from ...utils import conv as convUtils
from ...backpropextension import BackpropExtension
from ...extensions import BATCH_GRAD


class BatchGradConv2d(BackpropExtension):

    def __init__(self):
        super().__init__(
            torch.nn.Conv2d, BATCH_GRAD,
            req_inputs=[0], req_output=True
        )

    def apply(self, module, grad_input, grad_output):
        if module.bias is not None and module.bias.requires_grad:
            module.bias.grad_batch = self.bias_grad_batch(module, grad_output)
        if module.weight.requires_grad:
            module.weight.grad_batch = self.weight_grad_batch(module, grad_output)

    def bias_grad_batch(self, module, grad_output):
        """Compute bias batch gradients.

        - The batchwise gradient of a `torch.nn.Linear` layer is given
          by the gradient with respect to the output, `grad_output`
        - In `torch.nn.Conv2d` one bias element is added to all spatial locations
          of an output channel. Consequently, `grad_output` has to be summed
          over the spatial dimensions
        """
        return grad_output[0].sum(3).sum(2)

    def weight_grad_batch(self, module, grad_output):
        """Compute weight batch gradients.

        The convolution layer applies

        Y = W * X

        (neglecting the bias) to a single sample X, where W denotes the
        matrix view of the kernel, Y is a view of the output and X denotes
        the unfolded input matrix.

        Note on shapes/dims:
        --------------------
        - original input x: (batch, in_channels, x_dim, y_dim)
        - original kernel w: (out_channels, in_channels, k_x, k_y)
        - im2col input X: (batch, num_patches, in_channels * k_x * k_y)
        - kernel matrix W: (out_channels, in_channels *  k_x * k_y)
        - matmul result Y: (batch, out_channels, num_patches)
                       = (batch, out_channels, x_out_dim * y_out_dim)
        - col2im output y: (batch, out_channels, x_out_dim, y_out_dim)

        Forward pass: (pseudo)
        -------------
        X = unfold(x) = im2col(x)
        W = view(w)
        Y[b,i,j] = W[i,m] *  X[b,m,j]
        y = view(Y)

        Backward pass: (pseudo)
        --------------
        Given: dE / dy          (same shape as y)
        dE / dY = view(dE / dy) (same shape as Y)

        dE / dW[b,k,l]          (batch-wise gradient)
        dE / dW[b,k,l] = (dY[b,i,j] / dW[k,l]) * dE / dY[b,i,j]
                     = delta(i,k) * delta(m,l) * X[b,m,j] * dE / dY[b,i,j]
                     = delta(m,l) * X[b,m,j] * dE / dY[b,k,j]

        Result:
        -------
        dE / dw = view(dE / dW)
        """
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, grad_output[0], module
        )
        batch = module.input0.size(0)
        dE_dw_shape = (batch, ) + module.weight.size()
        dE_dW = einsum('bml,bkl->bmk', (dE_dY, X))
        return dE_dW.view(dE_dw_shape)


EXTENSIONS = [BatchGradConv2d()]
