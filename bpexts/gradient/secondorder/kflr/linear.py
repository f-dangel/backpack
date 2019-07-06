"""Factors for KFLR."""

from .kflrbase import KFLRBase
from ...derivatives.linear import LinearDerivatives
from ...context import CTX


class KFLRLinear(KFLRBase, LinearDerivatives):
    def __init__(self):
        # NOTE: Bias and weights treated jointly in KFLR, save in weights
        super().__init__(params=["weight"])

    def weight(self, module, grad_input, grad_output):
        # Naming of Kronecker factors: Equation (20) of the paper
        return (self.G(self, module, grad_input, grad_output),
                self.Q(self, module, grad_input, grad_output))

    def G(self, module, grad_input, grad_output):
        kflr_sqrt_ggn_out = self.get_sqrt_matrix()
        batch = kflr_sqrt_ggn_out.size(0)
        # NOTE: Normalization by batch size is already in the sqrt
        return einsum('bic,bjc->ij' (kflr_sqrt_ggn_out, kflr_sqrt_ggn_out))

    def Q(self, module, grad_input, grad_output):
        # append ones for the bias
        batch = module.input0.size(0)
        ones = torch.ones(
            batch, module.out_features, device=module.input0.device)
        input = torch.cat((module.input0, ones), dim=1)
        return einsum('bi,bj-> ij', (input, input)) / batch


EXTENSIONS = [KFLRLinear()]
