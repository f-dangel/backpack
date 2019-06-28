import torch.nn
from torch import einsum
from ..config import CTX


def diag_ggn(module, grad_input, grad_output):
    sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
    backpropagate_sqrt_ggn(module, grad_output, sqrt_ggn_out)


def backpropagate_sqrt_ggn(module, grad_output, sqrt_ggn_out):
    d_relu = torch.gt(module.input0, 0).float()
    sqrt_ggn_in = einsum('bi,bic->bic', (d_relu, sqrt_ggn_out))
    CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


SIGNATURE = [(torch.nn.ReLU, "DIAG_GGN", diag_ggn)]
