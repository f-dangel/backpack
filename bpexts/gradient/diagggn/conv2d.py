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
    raise NotImplementedError


def weight_diag_ggn(module, grad_output, sqrt_ggn_out):
    raise NotImplementedError


def backpropagate_sqrt_ggn(module, grad_output, sqrt_ggn_out):
    raise NotImplementedError


SIGNATURE = [(torch.nn.Conv2d, "DIAG_GGN", diag_ggn)]
