"""Factors for KFLR."""

from torch.nn import Linear
from ..context import CTX
from ...backpropextension import BackpropExtension
from ...extensions import KFLR
from ...jmp.linear import jac_mat_prod
from ....utils import einsum


# TODO second-order extension
class KFLRLinear(BackpropExtension):
    def __init__(self):
        # NOTE: Bias and weights treated jointly in KFLR, save in weights
        super().__init__(Linear, KFLR, params=["weight"])

    def weight(self, module, grad_input, grad_output):
        # Naming of Kronecker factors: Equation (20) of the paper
        return (self.G(self, module, grad_input, grad_output),
                self.Q(self, module, grad_input, grad_output))

    def G(self, module, grad_input, grad_output):
        kflr_sqrt_ggn_out = CTX._kflr_backpropagated_sqrt_ggn
        return einsum('bic,bjc->ij' (kflr_sqrt_ggn_out, kflr_sqrt_ggn_out))

    def Q(self, module, grad_input, grad_output):
        # append ones for the bias
        ones = torch.ones(
            module.input0.size(0),
            module.out_features,
            device=module.input0.device)
        input = torch.cat((module.input0, ones), dim=1)
        return einsum('bi,bj-> ij', (input, input))

    def backpropagate(module, grad_input, grad_output):
        kflr_sqrt_ggn_out = CTX._kflr_backpropagated_sqrt_ggn
        kflr_sqrt_ggn_in = self.jac_mat_prod(module, grad_input, grad_output,
                                             kflr_sqrt_ggn_out)
        CTX._kflr_backpropagated_sqrt_ggn = kflr_sqrt_ggn_in


EXTENSIONS = [KFLRLinear()]
