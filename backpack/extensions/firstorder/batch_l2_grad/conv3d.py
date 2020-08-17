from torch import einsum

from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.utils import conv as convUtils


class BatchL2Conv3d(FirstOrderModuleExtension):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        C_axis = 1
        return (einsum("ncdhw->nc", g_out[0]) ** 2).sum(C_axis)

    def weight(self, ext, module, g_inp, g_out, backproped):
        X, dE_dY = convUtils.get_conv3d_weight_gradient_factors(
            module.input0, g_out[0], module
        )
        return einsum("nmi,nki,nmj,nkj->n", (dE_dY, X, dE_dY, X))
