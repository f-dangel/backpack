from math import sqrt
from torch import diag_embed, ones_like, softmax, sqrt as torchsqrt, diag
from torch.nn import CrossEntropyLoss
from ...utils import einsum
from .basederivatives import BaseDerivatives


class CrossEntropyLossDerivatives(BaseDerivatives):
    def get_module(self):
        return CrossEntropyLoss

    def sqrt_hessian(self, module, grad_input, grad_output):
        probs = self.get_probs(module)
        tau = torchsqrt(probs)
        Id = diag_embed(ones_like(probs))
        Id_tautau = Id - einsum('ni,nj->nij', tau, tau)
        sqrt_H = einsum('ni,nij->nij', tau, Id_tautau)

        if module.reduction is "mean":
            sqrt_H /= sqrt(module.input0.shape[0])

        return sqrt_H

    def sum_hessian(self, module, grad_input, grad_output):
        probs = self.get_probs(module)
        sum_H = diag(probs.sum(0)) - einsum('bi,bj->ij', (probs, probs))

        if module.reduction is "mean":
            sum_H /= module.input0.shape[0]

        return sum_H

    def get_probs(self, module):
        return softmax(module.input0, dim=1)
