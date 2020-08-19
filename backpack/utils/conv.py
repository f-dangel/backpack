import torch
from torch import einsum
from torch.nn import Unfold
from torch.nn.functional import conv1d, conv2d, conv3d

from backpack.utils.ein import eingroup


def unfold_func(module):
    return Unfold(
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )


def get_conv1d_weight_gradient_factors(input, grad_out, module):
    # shape [N, C_in * K_x, L_out]
    X = unfold_by_conv(input, module)
    return X, grad_out


def get_weight_gradient_factors(input, grad_out, module):
    # shape [N, C_in * K_x * K_y, H_out * W_out]
    X = unfold_func(module)(input)
    dE_dY = eingroup("n,c,h,w->n,c,hw", grad_out)
    return X, dE_dY


def get_conv3d_weight_gradient_factors(input, grad_out, module):
    # shape [N, C_in * K_x * K_y * K_z, D_out * H_out * W_out]
    X = unfold_by_conv(input, module)
    dE_dY = eingroup("n,c,d,h,w->n,c,dhw", grad_out)
    return X, dE_dY


def separate_channels_and_pixels(module, tensor):
    """Reshape (V, N, C, H, W) into (V, N, C, H * W)."""
    return eingroup("v,n,c,h,w->v,n,c,hw", tensor)


def extract_weight_diagonal(module, input, grad_output):
    """
    input must be the unfolded input to the convolution (see unfold_func)
    and grad_output the backpropagated gradient
    """
    grad_output_viewed = separate_channels_and_pixels(module, grad_output)
    AX = einsum("nkl,vnml->vnkm", (input, grad_output_viewed))
    weight_diagonal = (AX ** 2).sum([0, 1]).transpose(0, 1)
    return weight_diagonal.view_as(module.weight)


def extract_bias_diagonal(module, sqrt):
    """
    `sqrt` must be the backpropagated quantity for DiagH or DiagGGN(MC)
    """
    V_axis, N_axis = 0, 1
    bias_diagonal = (einsum("vnchw->vnc", sqrt) ** 2).sum([V_axis, N_axis])
    return bias_diagonal


def unfold_by_conv(input, module):
    """Return the unfolded input using convolution"""
    N, C_in = input.shape[0], input.shape[1]
    kernel_size = module.kernel_size
    kernel_size_numel = int(torch.prod(torch.Tensor(kernel_size)))

    def make_weight():
        weight = torch.zeros(kernel_size_numel, 1, *kernel_size)

        for i in range(kernel_size_numel):
            extraction = torch.zeros(kernel_size_numel)
            extraction[i] = 1.0
            weight[i] = extraction.reshape(1, *kernel_size)

        repeat = [C_in, 1] + [1 for _ in kernel_size]
        return weight.repeat(*repeat)

    def get_conv():
        functional_for_module_cls = {
            torch.nn.Conv1d: conv1d,
            torch.nn.Conv2d: conv2d,
            torch.nn.Conv3d: conv3d,
        }
        return functional_for_module_cls[module.__class__]

    conv = get_conv()
    unfold = conv(
        input,
        make_weight().to(input.device),
        bias=None,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=C_in,
    )

    return unfold.reshape(N, C_in * kernel_size_numel, -1)
