import torch.nn
from torch import einsum
from ..config import CTX


def diag_ggn(module, grad_input, grad_output):
    sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
    backpropagate_sqrt_ggn(module, grad_input, grad_output, sqrt_ggn_out)


def backpropagate_sqrt_ggn(module, grad_input, grad_output, sqrt_ggn_out):
    scaling = 1 / (1 - module.p)
    mask = 1 - torch.eq(grad_input, 0.).float()
    d_dropout = mask * scaling
    sqrt_ggn_in = einsum('bo,boc->boc', d_dropout, sqrt_ggn_out)
    CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


SIGNATURE = [(torch.nn.Dropout, "DIAG_GGN", diag_ggn)]
