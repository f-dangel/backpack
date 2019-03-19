"""Extension of torch.nn.Linear for computing batch gradients."""

from torch.nn import Linear
from torch import einsum
from ..decorator import decorate


DecoratedLinear = decorate(Linear)


class SumGradSquared_Linear(DecoratedLinear):
    """Extended sum-of-gradients-squared backpropagation for torch.nn.Linear."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_exts_forward_pre_hook(self.store_input)
        self.register_exts_backward_hook(self.compute_sum_grad_squared)

    @staticmethod
    def store_input(module, input):
        """Pre forward hook saving layer input as buffer.

        Initialize module buffer `ìnput`.
        """
        if not len(input) == 1:
            raise ValueError('Cannot handle multi-input scenario')
        if not len(input[0].size()) == 2:
            raise ValueError('Expecting 2D input (batch, data)')

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
        by the gradient with respect to the layer's output.
        """
        return (grad_output.clone().detach()**2).sum(0)

    @staticmethod
    def compute_weight_sgs(module, grad_output):
        """Compute weight batch gradients from grad w.r.t layer outputs.

        The linear layer applies
        y = W x + b.
        to a single sample x, where W denotes the weight and b the bias.
        """
        if module.weight.requires_grad:
            return einsum('bi,bj->ij', (grad_output**2, module.input**2))

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
