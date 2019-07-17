import torch
import torch.nn
from ...context import CTX
from ...utils.utils import einsum
from ...core.derivatives.linear import LinearDerivatives, LinearConcatDerivatives
from .diaghbase import DiagHBase


class DiagHLinear(DiagHBase, LinearDerivatives):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    # TODO: Reuse code in ..diaggn.linear to extract the diagonal
    def bias(self, module, grad_input, grad_output):
        sqrt_h_outs = CTX._backpropagated_sqrt_h
        sqrt_h_outs_signs = CTX._backpropagated_sqrt_h_signs
        h_diag = torch.zeros_like(module.bias)
        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(sign * einsum('bic->i', (h_sqrt**2, )))
        return h_diag

    # TODO: Reuse code in ..diaggn.linear to extract the diagonal
    def weight(self, module, grad_input, grad_output):
        sqrt_h_outs = CTX._backpropagated_sqrt_h
        sqrt_h_outs_signs = CTX._backpropagated_sqrt_h_signs
        h_diag = torch.zeros_like(module.weight)
        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(sign * einsum('bic,bj->ij',
                                      (h_sqrt**2, module.input0**2)))
        return h_diag


class DiagHLinearConcat(DiagHBase, LinearConcatDerivatives):
    def __init__(self):
        super().__init__(params=["weight"])

    # TODO: Reuse code in ..diaggn.linear to extract the diagonal
    def weight(self, module, grad_input, grad_output):
        sqrt_h_outs = CTX._backpropagated_sqrt_h
        sqrt_h_outs_signs = CTX._backpropagated_sqrt_h_signs
        h_diag = torch.zeros_like(module.weight)

        input = module.homogeneous_input()

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(sign * einsum('bic,bj->ij', (h_sqrt**2, input**2)))
        return h_diag


EXTENSIONS = [DiagHLinear(), DiagHLinearConcat()]
