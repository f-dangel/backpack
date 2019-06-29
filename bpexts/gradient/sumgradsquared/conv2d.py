import torch.nn
from ...utils import einsum
from ..batchgrad.conv2d import bias_grad_batch, weight_grad_batch, unfold_func


def sum_grad_squared(module, grad_input, grad_output):
    """Compute sum of squared batch gradients of `torch.nn.Conv2d` parameters.

    - Store bias sum of squared gradients in `module.bias.sum_grad_squared`
    - Store kernel sum of squared gradients in `module.weight.sum_grad_squared`
    """
    if module.bias is not None and module.bias.requires_grad:
        module.bias.sum_grad_squared = bias_sum_grad_squared(
            module, grad_output)
    if module.weight.requires_grad:
        module.weight.sum_grad_squared = weight_sum_grad_squared(
            module, grad_output)


def bias_sum_grad_squared(module, grad_output):
    """Compute sum of squared batch gradients for bias."""
    return (bias_grad_batch(module, grad_output)**2).sum(0)


def weight_sum_grad_squared(module, grad_output):
    """Compute sum of squared batch gradients for kernel."""
    batch = module.input0.size(0)
    X = unfold_func(module)(module.input0)
    dE_dY = grad_output[0].view(batch, module.out_channels, -1)
    w_sgs = einsum('bml,bkl,bmi,bki->mk', (dE_dY, X, dE_dY, X))
    return w_sgs.view_as(module.weight)


SIGNATURE = [(torch.nn.Conv2d, "SUM_GRAD_SQUARED", sum_grad_squared)]
