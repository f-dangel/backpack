from torch import einsum

from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.utils import conv as convUtils


class BatchL2ConvND(FirstOrderModuleExtension):
    def __init__(self, N, params=None):
        super().__init__(params=params)
        self.N = N

    def bias(self, ext, module, g_inp, g_out, backproped):
        C_axis = 1
        return convUtils.get_bias_gradient_factors(g_out[0], C_axis, self.N)

    def weight(self, ext, module, g_inp, g_out, backproped):
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, g_out[0], module, self.N
        )
        return einsum("nmi,nki,nmj,nkj->n", (dE_dY, X, dE_dY, X))
