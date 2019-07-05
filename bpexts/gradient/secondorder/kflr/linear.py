"""Factors for KFLR."""

from .base import KFLRBase
from ...jacobians.linear import LinearJacobian
from ..context import CTX
from ....utils import einsum


class KFLRLinear(KFLRBase, LinearJacobian):
    def __init__(self):
        # NOTE: Bias and weights treated jointly in KFLR, save in weights
        super().__init__(params=["weight"])

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


EXTENSIONS = [KFLRLinear()]
