from torch import einsum, transpose

from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.utils import conv_transpose as convUtils


class SGSConvTranspose1d(FirstOrderModuleExtension):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        N_axis = 0
        return (einsum("ncl->nc", g_out[0]) ** 2).sum(N_axis)

    def weight(self, ext, module, g_inp, g_out, backproped):
        N_axis = 0

        X, dE_dY = convUtils.get_convtranspose1d_weight_gradient_factors(
            module.input0, g_out[0], module
        )
        C_in, C_out, K_X = module.weight.shape
        grad_batch = einsum("nml,nkl->nmk", (dE_dY, X))
        sgs = (grad_batch ** 2).sum(N_axis).reshape(C_out, C_in, K_X)
        return transpose(sgs, 0, 1)
