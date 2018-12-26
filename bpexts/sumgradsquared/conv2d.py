"""Extension of torch.nn.Conv2d for computing batch gradients."""

from torch.nn import (Conv2d, Unfold)
from torch import einsum
from numpy import prod
from ..decorator import decorate


# decorated torch.nn.Conv2d module
DecoratedConv2d = decorate(Conv2d)


class SumGradSquared_Conv2d(DecoratedConv2d):
    """Extended backpropagation for torch.nn.Conv2d."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unfold = Unfold(kernel_size=self.kernel_size,
                             dilation=self.dilation,
                             padding=self.padding,
                             stride=self.stride)
        self.register_forward_pre_hook(self.store_input)
        self.register_backward_hook(self.compute_sum_grad_squared)

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

    @staticmethod
    def compute_sum_grad_squared(module, grad_input, grad_output):
        """Backward hook for computing batch gradients.

        Store bias batch gradients in `module.bias.batch_grad` and
        weight batch gradients in `module.weight.batch_grad`.
        """
        if not len(grad_output) == 1:
            raise ValueError('Cannot handle multi-output scenario')
        if module.bias is not None and module.bias.requires_grad:
            module.bias.sum_grad_squared = module.compute_bias_sgs(module, grad_output[0])
        if module.weight.requires_grad:
            module.weight.sum_grad_squared = module.compute_weight_sgs(module, grad_output[0])

    @staticmethod
    def compute_bias_sgs(module, grad_output):
        """Compute bias batch gradients from grad w.r.t. layer outputs.

        The batchwise gradient of a linear layer is simply given
        by the gradient with respect to the layer's output, summed over
        the spatial dimensions (each number in the bias vector is added.
        to the spatial output of an entire channel).
        """
        return (grad_output.sum(3).sum(2)**2).sum(0)

    @staticmethod
    def compute_weight_sgs(module, grad_output):
        """Compute weight batch gradients from grad w.r.t layer outputs.

        The linear layer applies
        Y = W * X
        (neglecting the bias) to a single sample X, where W denotes the
        matrix view of the kernel, Y is a view of the output and X denotes
        the unfolded input matrix.
        """
        X = module.unfold(module.input)
        dE_dY = grad_output.view(grad_output.size()[0], module.out_channels, -1)

        return (einsum('bml,bkl->bmk', (dE_dY, X))**2).sum(0).view(module.weight.size())

    def clear_sum_grad_squared(self):
        """Delete batch gradients."""
        try:
            del self.weight.sum_grad_squared
        except AttributeError:
            pass
        try:
            del self.bias.sum_grad_squared
        except AttributeError:
            pass
