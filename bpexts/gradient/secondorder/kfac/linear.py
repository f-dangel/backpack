"""Factors for KFAC."""

import torch
from .kfacbase import KFACBase
from ...derivatives.linear import LinearDerivatives
from ....utils import einsum


class KFACLinear(KFACBase, LinearDerivatives):
    def __init__(self):
        # NOTE: Bias and weights treated jointly in KFAC, save in weights
        super().__init__(params=["weight"])

    def weight(self, module, grad_input, grad_output):
        r"""
        Note on the Kronecker factors regarding optimization:
        -----------------------------------------------------
        * vec denotes flattening in PyTorch (NOT column-stacking)
        * The concatenated parameter vector is [ (vec W)^T, b^T ]^T
        * In this flattening scheme, the curvature matrix C factorizes into
          C = G \otimes A
        """
        # Naming of Kronecker factors: Equation (1) of the paper
        return (self.A(module, grad_input, grad_output),
                self.G(module, grad_input, grad_output))

    # TODO: Refactor, same as in kflr.linear
    def A(self, module, grad_input, grad_output):
        # append ones for the bias
        batch = module.input0.size(0)
        ones = torch.ones(
            batch, module.out_features, device=module.input0.device)
        input = torch.cat((module.input0, ones), dim=1)
        return einsum('bi,bj-> ij', (input, input)) / batch

    def G(self, module, grad_input, grad_output):
        kfac_sqrt_mc_samples = self.get_from_ctx()
        samples = kfac_sqrt_mc_samples.size(2)
        # NOTE: Normalization by batch size is already in the sqrt
        return einsum('bim,bjm->ij',
                      (kfac_sqrt_mc_samples, kfac_sqrt_mc_samples)) / samples


EXTENSIONS = [KFACLinear()]
