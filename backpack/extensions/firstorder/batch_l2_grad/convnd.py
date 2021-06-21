from torch import einsum

from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.utils import conv as convUtils


class BatchL2ConvND(FirstOrderModuleExtension):
    def __init__(self, N, params=None):
        super().__init__(params=params)
        self.N = N

    # TODO Use bias Jacobian to compute `bias_gradient`
    def bias(self, ext, module, g_inp, g_out, backproped):
        spatial_dims = list(range(2, g_out[0].dim()))
        channel_dim = 1

        return g_out[0].sum(spatial_dims).pow_(2).sum(channel_dim)

    def weight(self, ext, module, g_inp, g_out, backproped):
        X, dE_dY = convUtils.get_weight_gradient_factors(
            module.input0, g_out[0], module, self.N
        )
        return einsum("nmi,nki,nmj,nkj->n", (dE_dY, X, dE_dY, X))
