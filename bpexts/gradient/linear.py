"""Extension of torch.nn.Linear for computing batch gradients."""

from torch.nn import Linear
from torch import einsum
from ..decorator import decorate


# decorated torch.nn.Linear module
DecoratedLinear = decorate(Linear)


class G_Linear(DecoratedLinear):
    """Extended gradient backpropagation for torch.nn.Linear."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_exts_forward_pre_hook(self.store_input)
        self.register_exts_backward_hook(self.compute_grad_batch)


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
        by the gradient with respect to the layer's output.
        """
        return grad_output.clone().detach()

    @staticmethod
    def compute_weight_grad_batch(module, grad_output):
        """Compute weight batch gradients from grad w.r.t layer outputs.

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
        if module.weight.requires_grad:
            batch_size = grad_output.size()[0]
            in_dim, out_dim = module.weight.size()
            weight_grad_batch = einsum('bi,bj->bij', (grad_output,
                                                      module.input))
            return weight_grad_batch.reshape(batch_size, in_dim, out_dim)
