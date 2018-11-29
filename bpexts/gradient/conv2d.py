"""Extension of torch.nn.Conv2d for computing batch gradients."""

from torch.nn import (Conv2d, Unfold)
from torch import (randn, einsum)
from numpy import prod
from ..decorator import decorate


# decorated torch.nn.Conv2d module
DecoratedConv2d = decorate(Conv2d)


class G_Conv2d(DecoratedConv2d):
    """Extended backpropagation for torch.nn.Conv2d."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unfold = Unfold(kernel_size=self.kernel_size,
                             dilation=self.dilation,
                             padding=self.padding,
                             stride=self.stride)
        self.register_forward_pre_hook(self.store_input)
        self.register_backward_hook(self.compute_grad_batch)

    @staticmethod
    def store_input(module, input):
        """Pre forward hook saving layer input as buffer.

        Initialize module buffer `ìnput`.
        """
        if not len(input) == 1:
            raise ValueError('Cannot handle multi-input scenario')
        if not len(input[0].size()) == 4:
            raise ValueError('Expecting 4D input (batch, channel, x, y)')
        module.register_exts_buffer('input', input[0].clone().detach())

    def forward(self, input):
        """Apply 2d convolution using unfold.

        More info on how to do this manually:
            https://discuss.pytorch.org/t/custom-convolution-dot-
            product/14992/7
        """
        out_size = self.output_size(input.size()[1:])
        out_shape = (-1, self.out_channels) + out_size
        # expand patches
        im2col = self.unfold(input)
        # perform convolution by matrix multiplication
        kernel_matrix = self.weight.view(self.out_channels, -1)
        # j: output image size, k: output channels
        convoluted = einsum('ki,bij->bkj', (kernel_matrix, im2col))
        # reshape into output image
        col2im = convoluted.view(out_shape)
        if self.bias is not None:
            bias = self.bias.view(1, -1, 1, 1)
            col2im.add_(bias.expand_as(col2im))
        return col2im

    def output_size(self, input_size):
        """Compute size of the output channels from random input.

        Taken from
            https://discuss.pytorch.org/t/convolution-and-pooling-
            layers-need-a-method-to-calculate-output-size/21895/2
        """
        x = randn(input_size).unsqueeze(0)
        output = super().forward(x)
        return output.size()[2:]

    @staticmethod
    def compute_grad_batch(module, grad_input, grad_output):
        """Backward hook for computing batch gradients.

        Store bias batch gradients in `module.bias.batch_grad` and
        weight batch gradients in `module.weight.batch_grad`.
        """
        if not len(grad_output) == 1:
            raise ValueError('Cannot handle multi-output scenario')
        if module.bias is not None:
            if module.bias.requires_grad:
                module.bias.grad_batch = module.compute_bias_grad_batch(
                    module, grad_output[0])
        if module.weight.requires_grad:
            module.weight.grad_batch = module.compute_weight_grad_batch(
                    module, grad_output[0])

    @staticmethod
    def compute_bias_grad_batch(module, grad_output):
        """Compute bias batch gradients from grad w.r.t. layer outputs.

        The batchwise gradient of a linear layer is simply given
        by the gradient with respect to the layer's output, summed over
        the spatial dimensions (each number in the bias vector is added.
        to the spatial output of an entire channel).
        """
        return grad_output.sum(3).sum(2)

    @staticmethod
    def compute_weight_grad_batch(module, grad_output):
        """Compute weight batch gradients from grad w.r.t layer outputs.

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
        batch_size = grad_output.size()[0]
        dE_dw_shape = (batch_size,) + module.weight.size()
        # expand patches
        X = module.unfold(module.input)
        # view of matmul result batch gradients
        dE_dY = grad_output.view(batch_size, module.out_channels, -1)
        # weight batch gradients dE/dW
        dE_dW = einsum('blj,bkj->bkl', (X, dE_dY))
        # reshape dE/dW into dE/dw
        return dE_dW.view(dE_dw_shape)
