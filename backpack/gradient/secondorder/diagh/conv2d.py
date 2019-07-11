import torch
import torch.nn
from ...context import CTX
from ....utils import einsum
from ...utils import conv as convUtils
from ....core.derivatives.conv2d import Conv2DDerivatives
from .diaghbase import DiagHBase
DETACH_INPUTS = True


class DiagHConv2d(DiagHBase, Conv2DDerivatives):
    def __init__(self):
        super().__init__(params=["bias", "weight"])

    # TODO: Reuse code in ..diaggn.conv2d to extract the diagonal
    def bias(self, module, grad_input, grad_output):
        sqrt_h_outs = CTX._backpropagated_sqrt_h
        sqrt_h_outs_signs = CTX._backpropagated_sqrt_h_signs
        h_diag = torch.zeros_like(module.bias)
        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_sqrt_view = convUtils.separate_channels_and_pixels(
                module, h_sqrt)
            h_diag.add_(sign * einsum('bijc,bikc->i',
                                      (h_sqrt_view, h_sqrt_view)))
        return h_diag

    # TODO: Reuse code in ..diaggn.conv2d to extract the diagonal
    def weight(self, module, grad_input, grad_output):
        sqrt_h_outs = CTX._backpropagated_sqrt_h
        sqrt_h_outs_signs = CTX._backpropagated_sqrt_h_signs
        X = convUtils.unfold_func(module)(module.input0).unsqueeze(0)
        h_diag = torch.zeros_like(module.weight)

        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            num_classes = h_sqrt.size(2)
            X_repeated = X.expand(num_classes, -1, -1, -1)
            h_sqrt_view = convUtils.separate_channels_and_pixels(
                module, h_sqrt)
            h_diag.add_(
                einsum('bmlc,cbkl,bmic,cbki->mk',
                       (h_sqrt_view, X_repeated, h_sqrt_view,
                        X_repeated)).view_as(module.weight))
        return h_diag


EXTENSIONS = [DiagHConv2d()]
