import torch
import torch.nn.functional as F
from torch import einsum
from ..config import CTX


def diag_ggn(module, grad_output):
    sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
    if module.input0.requires_grad:
        backpropagate_sqrt_ggn(module, grad_output, sqrt_ggn_out)


def backpropagate_sqrt_ggn(module, grad_output, sqrt_ggn_out):
    probs = F.softmax(module.input0, dim=1)
    tau = torch.sqrt(probs)
    Id = torch.diag_embed(torch.ones_like(probs))
    Id_tautau = Id - einsum('ni,nj->nij', tau, tau)
    sqrt_ggn_in = einsum('ni,nij->nij', tau, Id_tautau)
    CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


SIGNATURE = [(torch.nn.CrossEntropyLoss, "DIAG_GGN", diag_ggn)]
