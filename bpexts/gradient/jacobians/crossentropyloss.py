from math import sqrt
from torch import diag_embed, ones_like, softmax, sqrt as torchsqrt
from torch.nn import CrossEntropyLoss
from ...utils import einsum
from .basejacobian import BaseJacobian


class CrossEntropyLossJacobian(BaseJacobian):

    def get_module(self):
        return CrossEntropyLoss

    def sqrt_hessian(self, module, grad_input, grad_output):
        probs = softmax(module.input0, dim=1)
        tau = torchsqrt(probs)
        Id = diag_embed(ones_like(probs))
        Id_tautau = Id - einsum('ni,nj->nij', tau, tau)
        sqrt_H = einsum('ni,nij->nij', tau, Id_tautau)

        if module.reduction is "mean":
            sqrt_H /= sqrt(module.input0.shape[0])

        return sqrt_H
