from torch.nn.functional import softmax
from torch.nn import MSELoss, CrossEntropyLoss
from torch import diag_embed, ones_like
from torch import sqrt as torchsqrt
from math import sqrt
from ...utils import einsum
from ..context import CTX
from .base import DiagGGNBase


class DiagGGNMSELoss(DiagGGNBase):

    def get_module(self):
        return MSELoss

    def backpropagate(self, module, grad_input, grad_output):
        if not len(module.input0.shape) == 2:
            raise ValueError("Only 2D inputs are currently supported for MSELoss.")

        sqrt_ggn_in = diag_embed(sqrt(2) * ones_like(module.input0))

        if module.reduction is "mean":
            sqrt_ggn_in /= sqrt(module.input0.shape[0])

        CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


class DiagGGNCrossEntropyLoss(DiagGGNBase):

    def get_module(self):
        return CrossEntropyLoss

    def backpropagate(self, module, grad_input, grad_output):
        probs = softmax(module.input0, dim=1)
        tau = torchsqrt(probs)
        Id = diag_embed(ones_like(probs))
        Id_tautau = Id - einsum('ni,nj->nij', tau, tau)
        sqrt_ggn_in = einsum('ni,nij->nij', tau, Id_tautau)

        if module.reduction is "mean":
            sqrt_ggn_in /= sqrt(module.input0.shape[0])

        CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


EXTENSIONS = [
    DiagGGNCrossEntropyLoss(),
    DiagGGNMSELoss()
]
