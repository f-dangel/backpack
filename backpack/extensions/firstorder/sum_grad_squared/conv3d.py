from torch import einsum

from backpack.extensions.firstorder.base import FirstOrderModuleExtension
from backpack.utils import conv as convUtils


class SGSConv3d(FirstOrderModuleExtension):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, ext, module, g_inp, g_out, backproped):
        N_axis = 0
        return (einsum("ncdhw->nc", g_out[0]) ** 2).sum(N_axis)

    def weight(self, ext, module, g_inp, g_out, backproped):
        N_axis = 0
        X, dE_dY = convUtils.get_conv3d_weight_gradient_factors(
            module.input0, g_out[0], module
        )
        grad_batch = einsum("nml,nkl->nmk", (dE_dY, X))
        return (grad_batch ** 2).sum(N_axis).view_as(module.weight)
