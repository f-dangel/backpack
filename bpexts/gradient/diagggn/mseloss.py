import torch
import math
from ..config import CTX


def diag_ggn(module, grad_input, grad_output):
    backpropagate_sqrt_ggn(module)


def backpropagate_sqrt_ggn(module):
    if not len(module.input0.shape) == 2:
        raise ValueError("Only 2D inputs are currently supported for MSELoss.")

    sqrt_ggn_in = torch.diag_embed(math.sqrt(2) * torch.ones_like(module.input0))

    if module.reduction is "mean":
        sqrt_ggn_in /= math.sqrt(module.input0.shape[0])

    CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


SIGNATURE = [(torch.nn.MSELoss, "DIAG_GGN", diag_ggn)]
