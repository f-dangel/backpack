"""NOTE: KFLR for convolution layers is not defined. However, the only
difference to KFAC is the estimation of curvature by backpropagation of
curvature (KFLR) instead of MC sampling (KFAC). We therefore use the same
approximation as in the KFAC paper for convolutions (KFC) to obtain curvature
estimated for the convolution layer.
"""

from ...utils import conv as convUtils
from ...derivatives.conv2d import Conv2DDerivatives
from ....utils import einsum
from .kflrbase import KFLRBase


class KFLRConv2d(KFLRBase, Conv2DDerivatives):
    def __init__(self):
        # NOTE: Bias and weights treated jointly in KFLR, save in weights
        super().__init__(params=["weight"])

    def weight(self, module, grad_input, grad_output):
        # Naming of Kronecker factors: Equation (25) of the KFC paper
        return (self.Omega(self, module, grad_input, grad_output),
                self.Gamma(self, module, grad_input, grad_output))

    def Omega(self, module, grad_input, grad_output):
        # unfolded input
        X = convUtils.unfold_func(module)(module.input0).unsqueeze(0)
        # add ones for the bias terms
        ones = torch.ones(module.input0.size(0), 1, X.size(2), device=X.device)
        X_expanded = torch.cat((X, ones), dim=1)

        batch = X.size(0)
        return einsum('bik,bjk->ij', (X_expanded, X_expanded)) / batch

    def Gamma(self, module, grad_input, grad_output):
        kflr_sqrt_ggn_out = self.get_from_ctx()
        kflr_sqrt_ggn = convUtils.separate_channels_and_pixels(
            module, kflr_sqrt_ggn_out)
        # NOTE: Normalization might be different from KFC
        # NOTE: Normalization by batch size is already in the sqrt
        return einsum('bijc,blkc->il', (sqrt_ggn, sqrt_ggn))


EXTENSIONS = [KFLRConv2d()]
