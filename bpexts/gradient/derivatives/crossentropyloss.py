from math import sqrt
from torch import diag_embed, ones_like, softmax, sqrt as torchsqrt
from torch.nn import CrossEntropyLoss
from ...utils import einsum
from .basederivatives import BaseDerivatives


class CrossEntropyLossDerivatives(BaseDerivatives):

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

    def sqrt_hessian_sampled(self, module, grad_input, grad_output):
        N, C = module.input0.shape
        M = self.MC_SAMPLES

        probs = softmax(module.input0, dim=1)  # [N, C]
        ys = multinomial(probs, M, replacement=True)  # [N, M]

        # Compute G : [N, C, M], such that
        # G[n, c, m] = d_out[n,c] CE(out[n], ys[n,m])
        raise NotImplementedError

        # X = probs.unsqueeze(-1)
        # X = X.repeat(1, 1, M)  # [N, C, M]
        # one_hot = one_hot_encode(ys, C)  # [N, M, C]
        # one_hot = one_hot.transpose(1, 2)  # [N, C, M]

        # G = X - one_hot.float()

        # if module.reduction is "mean":
        #     G /= sqrt(module.input0.shape[0])

        # return -G, one_hot
