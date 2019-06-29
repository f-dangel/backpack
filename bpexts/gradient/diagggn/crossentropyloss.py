import torch
import math
import torch.nn.functional as F
from ..config import CTX
from ...utils import einsum


def diag_ggn(module, grad_input, grad_output):
    backpropagate_sqrt_ggn(module)


def backpropagate_sqrt_ggn(module):
    probs = F.softmax(module.input0, dim=1)
    tau = torch.sqrt(probs)
    Id = torch.diag_embed(torch.ones_like(probs))
    Id_tautau = Id - einsum('ni,nj->nij', tau, tau)
    sqrt_ggn_in = einsum('ni,nij->nij', tau, Id_tautau)

    if module.reduction is "mean":
        sqrt_ggn_in /= math.sqrt(module.input0.shape[0])

    CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


SIGNATURE = [(torch.nn.CrossEntropyLoss, "DIAG_GGN", diag_ggn)]
