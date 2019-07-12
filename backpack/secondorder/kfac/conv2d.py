import torch
from ...utils import conv as convUtils
from ...core.derivatives.conv2d import Conv2DDerivatives
from ...utils.utils import einsum
from .kfacbase import KFACBase


class KFACConv2d(KFACBase, Conv2DDerivatives):
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
          C = Gamma \otimes Omega
        """
        # Naming of Kronecker factors: Equation (25) of the KFC paper
        return (self.Omega(module, grad_input, grad_output),
                self.Gamma(module, grad_input, grad_output))

    def Omega(self, module, grad_input, grad_output):
        # unfolded input
        X = convUtils.unfold_func(module)(module.input0)
        # add ones for the bias terms
        ones = torch.ones(module.input0.size(0), 1, X.size(2), device=X.device)
        X_expanded = torch.cat((X, ones), dim=1)

        batch = X.size(0)
        return einsum('bik,bjk->ij', (X_expanded, X_expanded)) / batch

    def Gamma(self, module, grad_input, grad_output):
        kfac_sqrt_mc_samples = self.get_mat_from_ctx()
        samples = kfac_sqrt_mc_samples.size(2)
        kfac_sqrt_mc = convUtils.separate_channels_and_pixels(
            module, kfac_sqrt_mc_samples)
        # NOTE: Normalization might be different from KFC
        # NOTE: Normalization by batch size is already in the sqrt
        kfac_sqrt_mc = einsum('bijc->bic', (kfac_sqrt_mc, ))
        return einsum('bic,blc->il', (kfac_sqrt_mc, kfac_sqrt_mc)) / samples


EXTENSIONS = [KFACConv2d()]
