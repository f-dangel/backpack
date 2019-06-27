import torch.nn
from torch import einsum
from ..config import CTX


def diag_ggn(module, grad_output):
    sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
    if module.bias is not None and module.bias.requires_grad:
        module.bias.diag_ggn = bias_diag_ggn(module, grad_output)
    if module.weight.requires_grad:
        module.weight.diag_ggn = weight_diag_ggn(module, grad_output)
    if module.input0.requires_grad:
        backpropagate_sqrt_ggn(module, grad_output, sqrt_ggn_out)


def bias_diag_ggn(module, grad_output, sqrt_ggn_out):
    sqrt_ggn_bias = sqrt_ggn_out
    return einsum('bic,bic->i', (sqrt_ggn_bias, sqrt_ggn_bias))


def weight_diag_ggn(module, grad_output, sqrt_ggn_out):
    sqrt_ggn_weight = einsum('bic,bj->bijc', (sqrt_ggn_out, module.input0))
    return einsum('bijc,bijc->ij', (sqrt_ggn_weight, sqrt_ggn_weight))


def backpropagate_sqrt_ggn(module, grad_output, sqrt_ggn_out):
    sqrt_ggn_in = einsum('ij,bic->bjc', (module.weight, sqrt_ggn_out))
    CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


SIGNATURE = [(torch.nn.Linear, "DIAG_GGN", diag_ggn)]
