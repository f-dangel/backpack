import torch.nn
from torch import einsum
from ..config import CTX


def diag_ggn(module, grad_output):
    sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
    backpropagate_sqrt_ggn(module, grad_output, sqrt_ggn_out)


def backpropagate_sqrt_ggn(module, grad_output, sqrt_ggn_out):
    d_sigma = module.output * (1. - module.output)
    sqrt_ggn_in = einsum('bi,bic->bic', (d_sigma, sqrt_ggn_out))
    CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


SIGNATURE = [(torch.nn.Sigmoid, "DIAG_GGN", diag_ggn)]
