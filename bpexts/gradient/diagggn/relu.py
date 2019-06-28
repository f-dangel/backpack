import torch.nn
from ..config import CTX
from ..jmp.relu import jac_mat_prod


def diag_ggn(module, grad_input, grad_output):
    sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
    sqrt_ggn_in = jac_mat_prod(module, grad_input, grad_output, sqrt_ggn_out)
    CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


SIGNATURE = [(torch.nn.ReLU, "DIAG_GGN", diag_ggn)]
