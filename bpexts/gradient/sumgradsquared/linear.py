import torch.nn
from ...utils import einsum


def sum_grad_squared(module, grad_input, grad_output):
    """Compute sum of squared batch gradients of `torch.nn.Linear` parameters."""
    if module.bias is not None and module.bias.requires_grad:
        module.bias.sum_grad_squared = bias_sum_grad_squared(
            module, grad_output)
    if module.weight.requires_grad:
        module.weight.sum_grad_squared = weight_sum_grad_squared(
            module, grad_output)


def bias_sum_grad_squared(module, grad_output):
    """Compute sum of squared batch gradients for bias."""
    return (grad_output[0]**2).sum(0)


def weight_sum_grad_squared(module, grad_output):
    """Compute sum of squared batch gradients for weight."""
    w_sgs = einsum('bi,bj->ij', (grad_output[0]**2, module.input0**2))
    return w_sgs.view(module.out_features, module.in_features)


SIGNATURE = [(torch.nn.Linear, "SUM_GRAD_SQUARED", sum_grad_squared)]
