from backpack.core.derivatives.linear import LinearDerivatives
from backpack.utils.einsum import einsum
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNLinear(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        return einsum("vno->o", backproped ** 2)

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        return einsum("vno,ni->oi", (backproped ** 2, module.input0 ** 2))
