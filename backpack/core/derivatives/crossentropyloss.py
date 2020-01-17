from math import sqrt

from torch import diag, diag_embed, multinomial, ones_like, softmax
from torch import sqrt as torchsqrt
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot

from backpack.utils.einsum import einsum
from backpack.core.derivatives.basederivatives import BaseDerivatives
from backpack.core.derivatives.utils import (
    hessian_old_shape_convention,
    hessian_matrix_product_accept_vectors,
)


class CrossEntropyLossDerivatives(BaseDerivatives):
    def get_module(self):
        return CrossEntropyLoss

    # TODO: Convert [N, C, V] to  new convention [V, N, C]
    @hessian_old_shape_convention
    def sqrt_hessian(self, module, g_inp, g_out):
        probs = self.get_probs(module)
        tau = torchsqrt(probs)
        Id = diag_embed(ones_like(probs))
        Id_tautau = Id - einsum("ni,nj->nij", tau, tau)
        sqrt_H = einsum("ni,nij->nij", tau, Id_tautau)

        if module.reduction == "mean":
            N = module.input0.shape[0]
            sqrt_H /= sqrt(N)

        return sqrt_H

    # TODO: Convert [N, C, V] to  new convention [V, N, C]
    @hessian_old_shape_convention
    def sqrt_hessian_sampled(self, module, g_inp, g_out):
        M = self.MC_SAMPLES
        C = module.input0.shape[1]

        probs = self.get_probs(module)
        probs_unsqueezed = probs.unsqueeze(-1).repeat(1, 1, M)

        classes = one_hot(multinomial(probs, M, replacement=True), num_classes=C)
        classes = classes.transpose(1, 2).float()

        sqrt_mc_h = (probs_unsqueezed - classes) / sqrt(M)

        if module.reduction == "mean":
            N = module.input0.shape[0]
            sqrt_mc_h /= sqrt(N)

        return sqrt_mc_h

    def sum_hessian(self, module, g_inp, g_out):
        probs = self.get_probs(module)
        sum_H = diag(probs.sum(0)) - einsum("bi,bj->ij", (probs, probs))

        if module.reduction == "mean":
            N = module.input0.shape[0]
            sum_H /= N

        return sum_H

    @hessian_matrix_product_accept_vectors
    def hessian_matrix_product(self, module, g_inp, g_out):
        """Multiplication of the input Hessian with a matrix."""
        probs = self.get_probs(module)

        def hmp(mat):
            Hmat = einsum("bi,cbi->cbi", (probs, mat)) - einsum(
                "bi,bj,cbj->cbi", (probs, probs, mat)
            )

            if module.reduction == "mean":
                N = module.input0.shape[0]
                Hmat /= N

            return Hmat

        return hmp

    def hessian_is_psd(self):
        return True

    def get_probs(self, module):
        return softmax(module.input0, dim=1)
