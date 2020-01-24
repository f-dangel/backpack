from math import sqrt

from torch import diag, diag_embed, multinomial, ones_like, softmax
from torch import sqrt as torchsqrt
from torch.nn import CrossEntropyLoss
from torch.nn.functional import one_hot

from backpack.utils.einsum import einsum
from backpack.core.derivatives.basederivatives import BaseLossDerivatives


class CrossEntropyLossDerivatives(BaseLossDerivatives):
    def get_module(self):
        return CrossEntropyLoss

    def _sqrt_hessian(self, module, g_inp, g_out):
        probs = self.get_probs(module)
        tau = torchsqrt(probs)
        V_dim, C_dim = 0, 2
        Id = diag_embed(ones_like(probs), dim1=V_dim, dim2=C_dim)
        Id_tautau = Id - einsum("nv,nc->vnc", tau, tau)
        sqrt_H = einsum("nc,vnc->vnc", tau, Id_tautau)

        if module.reduction == "mean":
            N = module.input0.shape[0]
            sqrt_H /= sqrt(N)

        return sqrt_H

    def _sqrt_hessian_sampled(self, module, g_inp, g_out, mc_samples=1):
        M = mc_samples
        C = module.input0.shape[1]

        probs = self.get_probs(module)
        V_dim = 0
        probs_unsqueezed = probs.unsqueeze(V_dim).repeat(M, 1, 1)

        multi = multinomial(probs, M, replacement=True)
        classes = one_hot(multi, num_classes=C)
        classes = einsum("nvc->vnc", classes).float()

        sqrt_mc_h = (probs_unsqueezed - classes) / sqrt(M)

        if module.reduction == "mean":
            N = module.input0.shape[0]
            sqrt_mc_h /= sqrt(N)

        return sqrt_mc_h

    def _sum_hessian(self, module, g_inp, g_out):
        probs = self.get_probs(module)
        sum_H = diag(probs.sum(0)) - einsum("bi,bj->ij", (probs, probs))

        if module.reduction == "mean":
            N = module.input0.shape[0]
            sum_H /= N

        return sum_H

    def _make_hessian_mat_prod(self, module, g_inp, g_out):
        """Multiplication of the input Hessian with a matrix."""
        probs = self.get_probs(module)

        def hessian_mat_prod(mat):
            Hmat = einsum("bi,cbi->cbi", (probs, mat)) - einsum(
                "bi,bj,cbj->cbi", (probs, probs, mat)
            )

            if module.reduction == "mean":
                N = module.input0.shape[0]
                Hmat /= N

            return Hmat

        return hessian_mat_prod

    def hessian_is_psd(self):
        return True

    def get_probs(self, module):
        return softmax(module.input0, dim=1)
