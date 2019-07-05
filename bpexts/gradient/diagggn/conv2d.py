import torch.nn
from ..context import CTX
from ..utils import conv as convUtils
from ..jmp.conv2d import jac_mat_prod
from ...utils import einsum
from ..backpropextension import BackpropExtension
from ..extensions import DIAG_GGN


class DiagGGNConv2d(BackpropExtension):

    def __init__(self):
        super().__init__(torch.nn.Conv2d, DIAG_GGN, params=["bias", "weight"])

    def bias(self, module, grad_output, sqrt_ggn_out):
        sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn_out)
        return einsum('bijc,bikc->i', (sqrt_ggn, sqrt_ggn))

    def weight(self, module, grad_output, sqrt_ggn_out):
        sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn_out)

        # unfolded input, repeated for each class
        num_classes = sqrt_ggn_out.size(2)
        X = convUtils.unfold_func(module)(module.input0).unsqueeze(0)
        X = X.expand(num_classes, -1, -1, -1)

        return einsum('bmlc,cbkl,bmic,cbki->mk',
                      (sqrt_ggn, X, sqrt_ggn, X)).view_as(module.weight)

    def backpropagate(self, module, grad_input, grad_output):
        sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
        sqrt_ggn_in = jac_mat_prod(module, grad_input, grad_output, sqrt_ggn_out)
        CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


EXTENSIONS = [DiagGGNConv2d()]
