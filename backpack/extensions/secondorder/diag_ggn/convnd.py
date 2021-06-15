from backpack.extensions.secondorder.diag_ggn.diag_ggn_base import DiagGGNBaseModule
from backpack.utils import conv as convUtils


class DiagGGNConvND(DiagGGNBaseModule):
    def bias(self, ext, module, grad_inp, grad_out, backproped):
        sqrt_ggn = backproped
        return convUtils.extract_bias_diagonal(module, sqrt_ggn, sum_batch=True)

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        X = convUtils.unfold_input(module, module.input0)
        weight_diag = convUtils.extract_weight_diagonal(
            module, X, backproped, sum_batch=True
        )
        return weight_diag


class BatchDiagGGNConvND(DiagGGNBaseModule):
    def bias(self, ext, module, grad_inp, grad_out, backproped):
        sqrt_ggn = backproped
        return convUtils.extract_bias_diagonal(module, sqrt_ggn, sum_batch=False)

    def weight(self, ext, module, grad_inp, grad_out, backproped):
        X = convUtils.unfold_input(module, module.input0)
        weight_diag = convUtils.extract_weight_diagonal(
            module, X, backproped, sum_batch=False
        )
        return weight_diag
