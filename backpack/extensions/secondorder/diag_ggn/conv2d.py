from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.utils import conv as convUtils
from backpack.utils.einsum import einsum

from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNConv2d(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=Conv2DDerivatives(), params=["bias", "weight"])

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        sqrt_ggn = einsum("vnchw->vnc", backproped)
        V_axis, N_axis = 0, 1
        return (sqrt_ggn ** 2).sum([V_axis, N_axis])

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        X = convUtils.unfold_func(module)(module.input0)
        weight_diag = convUtils.extract_weight_diagonal(module, X, backproped)
        return weight_diag
