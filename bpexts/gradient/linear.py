"""Extension of torch.nn.Linear for computing batch gradients."""

from torch.nn import Linear
from ..decorator import decorate


# decorated torch.nn.Linear module
DecoratedLinear = decorate(Linear)


class G_Linear(DecoratedLinear):
    """Extended gradient backpropagation for torch.nn.Linear."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_exts_backward_hook(G_Linear.compute_batch_grad)

    @staticmethod
    def compute_batch_grad(module, grad_input, grad_output):
        """Backward hook for computing batch gradients."""
        if not len(grad_output) == 1:
            raise ValueError('Cannot handle multi-output scenario')
        if module.bias is not None:
            module.compute_bias_batch_grad(module, grad_output[0])

    @staticmethod
    def compute_bias_batch_grad(module, grad_output):
        """Compute bias batch gradients from grad w.r.t. layer outputs.

        The batchwise gradient of a linear layer is simply given
        by the gradient with respect to the layer's output.
        """
        if module.bias.requires_grad:
            module.bias.grad_batch = grad_output.clone().detach()
