from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule
from backpack.utils import conv as convUtils


class DiagGGNConv2d(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=Conv2DDerivatives(), params=["bias", "weight"])

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        sqrt_ggn = backproped
        return convUtils.extract_bias_diagonal(module, sqrt_ggn)

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        X = convUtils.unfold_func(module)(module.input0)
        weight_diag = convUtils.extract_weight_diagonal(module, X, backproped)
        return weight_diag
