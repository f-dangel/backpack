from ...core.derivatives.linear import LinearDerivatives, LinearConcatDerivatives
from ...core.layers import LinearConcat
from ...utils.utils import einsum
from .diagggnbase import DiagGGNBase


class DiagGGNLinear(DiagGGNBase, LinearDerivatives):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        sqrt_ggn_bias = self.get_mat_from_ctx()
        return einsum('bic->i', (sqrt_ggn_bias**2, ))

    def weight(self, module, grad_input, grad_output):
        sqrt_ggn_out = self.get_mat_from_ctx()
        return einsum('bic,bj->ij', (sqrt_ggn_out**2, module.input0**2))


class DiagGGNLinearConcat(DiagGGNBase, LinearConcatDerivatives):
    def __init__(self):
        super().__init__(params=["weight"])

    def weight(self, module, grad_input, grad_output):
        sqrt_ggn_out = self.get_mat_from_ctx()

        input = module.homogeneous_input()

        return einsum('bic,bj->ij', (sqrt_ggn_out**2, input**2))


EXTENSIONS = [DiagGGNLinear(), DiagGGNLinearConcat()]
