from ..context import CTX
from ..jacobians.linear import LinearJacobian
from ...utils import einsum
from .diagggnbase import DiagGGNBase


class DiagGGNLinear(DiagGGNBase, LinearJacobian):

    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        sqrt_ggn_bias = CTX._backpropagated_sqrt_ggn
        return einsum('bic->i', (sqrt_ggn_bias**2, ))

    def weight(self, module, grad_input, grad_output):
        sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
        return einsum('bic,bj->ij', (sqrt_ggn_out**2, module.input0**2))


EXTENSIONS = [DiagGGNLinear()]
