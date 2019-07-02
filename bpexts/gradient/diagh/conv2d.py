import torch
import torch.nn
from ..context import CTX
from ...utils import einsum
from ..backpropextension import BackpropExtension
from ..jmp.conv2d import jac_mat_prod
from ..utils import unfold_func
from ..extensions import DIAG_H


class DiagHConv2d(BackpropExtension):
    def __init__(self):
        super().__init__(torch.nn.Conv2d, DIAG_H, req_inputs=[0])

    def apply(self, module, grad_input, grad_output):
        sqrt_h_outs = CTX._backpropagated_sqrt_h
        sqrt_h_outs_signs = CTX._backpropagated_sqrt_h_signs

        if module.bias is not None and module.bias.requires_grad:
            module.bias.diag_h = bias_diagH(module, sqrt_h_outs,
                                            sqrt_h_outs_signs)
        if module.weight.requires_grad:
            module.weight.diag_h = weight_diagH(module, sqrt_h_outs,
                                                sqrt_h_outs_signs)
        if module.input0.requires_grad:
            backpropagate_sqrt_h(module, grad_input, grad_output, sqrt_h_outs,
                                 sqrt_h_outs_signs)

    # TODO: Reuse code in ..diaggn.conv2d to extract the diagonal
    def bias_diagH(self, module, sqrt_h_outs, sqrt_h_outs_signs):
        h_diag = torch.zeros_like(module.bias)
        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_sqrt_view = self.separate_channels_and_pixels(module, h_sqrt)
            h_diag.add_(sign * einsum('bijc,bikc->i',
                                      (h_sqrt_view, h_sqrt_view)))
        return h_diag

    # TODO: Reuse code in ..diaggn.conv2d to extract the diagonal
    def weight_diagH(self, module, sqrt_h_outs, sqrt_h_outs_signs):
        # unfolded input, repeated for each class
        num_classes = sqrt_h_outs[0].size(2)
        X = unfold_func(module)(module.input0).unsqueeze(0).expand(
            num_classes, -1, -1, -1)

        h_diag = torch.zeros_like(module.weight)
        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_sqrt_view = self.separate_channels_and_pixels(module, h_sqrt)
            h_diag.add_(
                einsum('bmlc,cbkl,bmic,cbki->mk',
                       (q_sqrt_view, X, h_sqrt_view, X)).view_as(
                           module.weight))
        return h_diag

    # TODO: Reuse code in ..diaggn.conv2d
    def separate_channels_and_pixels(self, module, sqrt_ggn_out):
        """Reshape (batch, out_features, classes)
        into
                   (batch, out_channels, pixels, classes).
        """
        batch, channels, pixels, classes = (
            module.input0.size(0),
            module.out_channels,
            module.output_shape[2] * module.output_shape[3],
            sqrt_ggn_out.size(2),
        )
        return sqrt_ggn_out.view(batch, channels, pixels, classes)

    def backpropagate_sqrt_h(self, module, grad_input, grad_output,
                             sqrt_h_outs, sqrt_h_outs_signs):
        for i, sqrt_h in enumerate(sqrt_h_outs):
            sqrt_h_outs[i] = jac_mat_prod(module, grad_input, grad_output,
                                          sqrt_h)
        CTX._backpropagated_sqrt_h = sqrt_h_outs


EXTENSIONS = [DiagHConv2d()]
