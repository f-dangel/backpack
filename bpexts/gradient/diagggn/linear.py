import torch.nn
from torch import einsum
from ..config import CTX


def diag_ggn(module, grad_input, grad_output):
    sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
    if module.bias is not None and module.bias.requires_grad:
        module.bias.diag_ggn = bias_diag_ggn(module, grad_output, sqrt_ggn_out)
    if module.weight.requires_grad:
        module.weight.diag_ggn = weight_diag_ggn(module, grad_output, sqrt_ggn_out)

    backpropagate_sqrt_ggn(module, grad_output, sqrt_ggn_out)


def bias_diag_ggn(module, grad_output, sqrt_ggn_out):
    sqrt_ggn_bias = sqrt_ggn_out
    return einsum('bic->i', (sqrt_ggn_bias**2, ))


def weight_diag_ggn(module, grad_output, sqrt_ggn_out):
    return einsum('bic,bj->ij', (sqrt_ggn_out**2, module.input0**2))


def backpropagate_sqrt_ggn(module, grad_output, sqrt_ggn_out):
    sqrt_ggn_in = einsum('ij,bic->bjc', (module.weight, sqrt_ggn_out))
    CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


SIGNATURE = [(torch.nn.Linear, "DIAG_GGN", diag_ggn)]
