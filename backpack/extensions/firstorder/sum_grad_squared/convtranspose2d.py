from torch import einsum, transpose

from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.utils import conv_transpose as convUtils


class SGSConvTranspose2d(FirstOrderModuleExtension):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        C_axis = 0
        return (einsum("nchw->nc", g_out[0]) ** 2).sum(C_axis)

    def weight(self, ext, module, g_inp, g_out, backproped):
        C_axis = 0
        C_out_axis = 1
        X, dE_dY = convUtils.get_convtranspose2d_weight_gradient_factors(
            module.input0, g_out[0], module
        )
        C_in, C_out, K_X, K_Y = module.weight.shape
        d1 = einsum("nml,nkl->nmk", (dE_dY, X))
        d2 = (d1 ** 2).sum(C_axis).reshape(C_out, C_in, K_X, K_Y)
        return transpose(d2, C_axis, C_out_axis)
