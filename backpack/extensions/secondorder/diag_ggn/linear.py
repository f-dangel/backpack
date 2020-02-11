import backpack.utils.linear as LinUtils
from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule


class DiagGGNLinear(DiagGGNBaseModule):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    def bias(self, ext, module, grad_inp, grad_out, backproped):
        return LinUtils.extract_bias_diagonal(module, backproped)

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        return LinUtils.extract_weight_diagonal(module, backproped)
