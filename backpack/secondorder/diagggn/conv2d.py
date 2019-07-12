from ...utils import conv as convUtils
from ...core.derivatives.conv2d import Conv2DDerivatives
from ...utils.utils import einsum
from .diagggnbase import DiagGGNBase


class DiagGGNConv2d(DiagGGNBase, Conv2DDerivatives):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        sqrt_ggn_out = self.get_mat_from_ctx()
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn_out)

        return einsum('bijc,bikc->i', (sqrt_ggn, sqrt_ggn))

    def weight(self, module, grad_input, grad_output):
        sqrt_ggn_out = self.get_mat_from_ctx()
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn_out)

        X = convUtils.unfold_func(module)(module.input0)

        AX = einsum('bkl,bmlc->cbkm', (X, sqrt_ggn))
        AXAX = (AX**2).sum([0, 1]).transpose(0, 1)

        return AXAX.view_as(module.weight)


EXTENSIONS = [DiagGGNConv2d()]
