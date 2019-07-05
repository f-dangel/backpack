from ..context import CTX
from ..utils import conv as convUtils
from ..jacobians.conv2d import Conv2dJacobian
from ...utils import einsum
from .diagggnbase import DiagGGNBase


class DiagGGNConv2d(DiagGGNBase, Conv2dJacobian):

    def __init__(self):
        super().__init__(params=["bias", "weight"])

    def bias(self, module, grad_input, grad_output):
        sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn_out)
        return einsum('bijc,bikc->i', (sqrt_ggn, sqrt_ggn))

    def weight(self, module, grad_input, grad_output):
        sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn_out)

        # unfolded input, repeated for each class
        num_classes = sqrt_ggn_out.size(2)
        X = convUtils.unfold_func(module)(module.input0).unsqueeze(0)
        X = X.expand(num_classes, -1, -1, -1)

        return einsum('bmlc,cbkl,bmic,cbki->mk',
                      (sqrt_ggn, X, sqrt_ggn, X)).view_as(module.weight)


EXTENSIONS = [DiagGGNConv2d()]
