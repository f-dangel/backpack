"""Factors for KFAC."""

from .kfacbase import KFACBase
from ...derivatives.linear import LinearDerivatives


class KFACLinear(KFACBase, LinearDerivatives):
    def __init__(self):
        # NOTE: Bias and weights treated jointly in KFAC, save in weights
        super().__init__(params=["weight"])

    def weight(self, module, grad_input, grad_output):
        # Naming of Kronecker factors: Equation (1) of the paper
        return (self.G(self, module, grad_input, grad_output),
                self.A(self, module, grad_input, grad_output))

    def G(self, module, grad_input, grad_output):
        kfac_sqrt_mc_samples = self.get_from_ctx()
        batch = kfac_sqrt_mc_samples.size(0)
        # NOTE: Normalization by batch size is already in the sqrt
        # TODO: Additional normalization for more than one MC sample? (m>1)
        return einsum('bim,bjm->ij',
                      (kfac_sqrt_mc_samples, kfac_sqrt_mc_samples))

    # TODO: Refactor, same as in kflr.linear
    def A(self, module, grad_input, grad_output):
        # append ones for the bias
        batch = module.input0.size(0)
        ones = torch.ones(
            batch, module.out_features, device=module.input0.device)
        input = torch.cat((module.input0, ones), dim=1)
        return einsum('bi,bj-> ij', (input, input)) / batch


EXTENSIONS = [KFACLinear()]
