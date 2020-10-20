from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule
from backpack.utils import conv as convUtils
from backpack.utils import conv_transpose as convTransposeUtils


class DiagGGNConvBase(DiagGGNBaseModule):
    def __init__(self, derivatives, N, params=None, convtranspose=False):
        super().__init__(derivatives=derivatives, params=params)
        self.N = N
        if convtranspose == True:
            self.convUtils = convTransposeUtils
            self.unfold = convTransposeUtils.unfold_by_conv_transpose
        else:
            self.convUtils = convUtils
            self.unfold = convUtils.unfold_by_conv

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        sqrt_ggn = backproped
        return self.convUtils.extract_bias_diagonal(module, sqrt_ggn, self.N)

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        X = self.unfold(module.input0, module)
        weight_diag = self.convUtils.extract_weight_diagonal(
            module, X, backproped, self.N
        )
        return weight_diag
