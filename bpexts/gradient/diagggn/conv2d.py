import torch.nn
from ..context import CTX
from ..utils import conv as convUtils
from ..jmp.conv2d import jac_mat_prod
from ...utils import einsum
from ..backpropextension import BackpropExtension
from ..extensions import DIAG_GGN


class DiagGGNConv2d(BackpropExtension):

    def __init__(self):
        super().__init__(
            torch.nn.Conv2d, DIAG_GGN,
            req_inputs=[0], req_output=True
        )

    def apply(self, module, grad_input, grad_output):
        sqrt_ggn_out = CTX._backpropagated_sqrt_ggn

        if module.bias is not None and module.bias.requires_grad:
            module.bias.diag_ggn = self.bias_diag_ggn(module, grad_output, sqrt_ggn_out)
        if module.weight.requires_grad:
            module.weight.diag_ggn = self.weight_diag_ggn(module, grad_output,
                                                          sqrt_ggn_out)

        self.backpropagate_sqrt_ggn(module, grad_input, grad_output, sqrt_ggn_out)

    def bias_diag_ggn(self, module, grad_output, sqrt_ggn_out):
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn_out)
        return einsum('bijc,bikc->i', (sqrt_ggn, sqrt_ggn))

    def weight_diag_ggn(self, module, grad_output, sqrt_ggn_out):
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn_out)

        # unfolded input, repeated for each class
        num_classes = sqrt_ggn_out.size(2)
        X = convUtils.unfold_func(module)(module.input0).unsqueeze(0)
        X = X.expand(num_classes, -1, -1, -1)

        return einsum('bmlc,cbkl,bmic,cbki->mk',
                      (sqrt_ggn, X, sqrt_ggn, X)).view_as(module.weight)

    def backpropagate_sqrt_ggn(self, module, grad_input, grad_output, sqrt_ggn_out):
        sqrt_ggn_in = jac_mat_prod(module, grad_input, grad_output, sqrt_ggn_out)
        CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


EXTENSIONS = [DiagGGNConv2d()]
