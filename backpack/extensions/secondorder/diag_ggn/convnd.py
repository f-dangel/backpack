from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule
from backpack.utils import conv as convUtils


class DiagGGNConvND(DiagGGNBaseModule):
    def __init__(self, derivatives, N, params=None):
        super().__init__(derivatives=derivatives, params=params)
        self.N = N

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        sqrt_ggn = backproped
        return convUtils.extract_bias_diagonal(module, sqrt_ggn, self.N)

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        X = convUtils.unfold_by_conv(module.input0, module)
        weight_diag = convUtils.extract_weight_diagonal(module, X, backproped, self.N)
        return weight_diag
