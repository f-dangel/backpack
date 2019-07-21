import torch
import torch.nn
from backpack.utils.utils import einsum
from backpack.core.derivatives.linear import LinearDerivatives, LinearConcatDerivatives
from .diag_h_base import DiagHBaseModule


class DiagHLinear(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["bias", "weight"])

    # TODO: Reuse code in ..diaggn.linear to extract the diagonal
    def bias(self, ext, module, g_inp, g_out, backproped):
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]

        h_diag = torch.zeros_like(module.bias)
        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(sign * einsum('bic->i', (h_sqrt**2, )))
        return h_diag

    # TODO: Reuse code in ..diaggn.linear to extract the diagonal
    def weight(self, ext, module, g_inp, g_out, backproped):
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]

        h_diag = torch.zeros_like(module.weight)
        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(sign * einsum('bic,bj->ij',
                                      (h_sqrt**2, module.input0**2)))
        return h_diag


class DiagHLinearConcat(DiagHBaseModule):
    def __init__(self):
        super().__init__(derivatives=LinearConcatDerivatives(), params=["weight"])

    # TODO: Reuse code in ..diaggn.linear to extract the diagonal
    def weight(self, ext, module, g_inp, g_out, backproped):
        sqrt_h_outs = backproped["matrices"]
        sqrt_h_outs_signs = backproped["signs"]
        h_diag = torch.zeros_like(module.weight)

        input = module.homogeneous_input()

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(sign * einsum('bic,bj->ij', (h_sqrt**2, input**2)))
        return h_diag

