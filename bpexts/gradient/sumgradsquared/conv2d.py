import torch.nn
from torch import einsum
from ..batchgrad.conv2d import bias_grad_batch, weight_grad_batch


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
    # TODO: This could probably be optimized with einsum
    return (weight_grad_batch(module, grad_output)**2).sum(0)


SIGNATURE = [(torch.nn.Conv2d, "SUM_GRAD_SQUARED", sum_grad_squared)]
