from backpack.core.derivatives.linear import LinearDerivatives
from backpack.utils.einsum import einsum

from .diag_ggn_base import DiagGGNBaseModule


class DiagGGNLinear(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        return einsum("bic->i", (backproped ** 2,))

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        return einsum("bic,bj->ij", (backproped ** 2, module.input0 ** 2))
