import torch.nn
from torch import einsum
from ..config import CTX
from ..batchgrad.conv2d import unfold_func
from ..jmp.conv2d import jac_mat_prod


def diag_ggn(module, grad_input, grad_output):
    sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
    if module.bias is not None and module.bias.requires_grad:
        module.bias.diag_ggn = bias_diag_ggn(module, grad_output, sqrt_ggn_out)
    if module.weight.requires_grad:
        module.weight.diag_ggn = weight_diag_ggn(module, grad_output,
                                                 sqrt_ggn_out)

    backpropagate_sqrt_ggn(module, grad_input, grad_output, sqrt_ggn_out)


def bias_diag_ggn(module, grad_output, sqrt_ggn_out):
    sqrt_ggn = separate_channels_and_pixels(module, sqrt_ggn_out)
    return einsum('bijc,bikc->i', (sqrt_ggn, sqrt_ggn))


def weight_diag_ggn(module, grad_output, sqrt_ggn_out):
    sqrt_ggn = separate_channels_and_pixels(module, sqrt_ggn_out)

    # unfolded input, repeated for each class
    num_classes = sqrt_ggn_out.size(2)
    X = unfold_func(module)(module.input0).unsqueeze(0).expand(
        num_classes, -1, -1, -1)

    return einsum('bmlc,cbkl,bmic,cbki->mk',
                  (sqrt_ggn, X, sqrt_ggn, X)).view_as(module.weight)


def separate_channels_and_pixels(module, sqrt_ggn_out):
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


def backpropagate_sqrt_ggn(module, grad_input, grad_output, sqrt_ggn_out):
    sqrt_ggn_in = jac_mat_prod(module, grad_input, grad_output, sqrt_ggn_out)
    CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


SIGNATURE = [(torch.nn.Conv2d, "DIAG_GGN", diag_ggn)]
