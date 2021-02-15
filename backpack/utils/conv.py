import torch
from torch import einsum
from torch.nn import Unfold
from torch.nn.functional import conv1d, conv2d, conv3d

from einops import rearrange


def unfold_func(module):
    return Unfold(
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )


def get_weight_gradient_factors(input, grad_out, module, N):
    # shape [N, C_in * K_x * K_y, H_out * W_out]
    if N == 1:
        X = unfold_by_conv(module.input0, module)
    elif N == 2:
        X = unfold_func(module)(input)
    elif N == 3:
        X = unfold_by_conv(module.input0, module)
    else:
        raise ValueError("{}-dimensional Conv. is not implemented.".format(N))

    dE_dY = rearrange(grad_out, "n c ... -> n c (...)")
    return X, dE_dY


def get_bias_gradient_factors(gradient, C_axis, N):
    if N == 1:
        bias_gradient = (einsum("ncl->nc", gradient) ** 2).sum(C_axis)
    elif N == 2:
        bias_gradient = (einsum("nchw->nc", gradient) ** 2).sum(C_axis)
    elif N == 3:
        bias_gradient = (einsum("ncdhw->nc", gradient) ** 2).sum(C_axis)
    else:
        raise ValueError("{}-dimensional Conv. is not implemented.".format(N))
    return bias_gradient


def separate_channels_and_pixels(module, tensor):
    """Reshape (V, N, C, H, W) into (V, N, C, H * W)."""
    return rearrange(tensor, "v n c ... -> v n c (...)")


def extract_weight_diagonal(module, input, grad_output, N, sum_batch=True):
    """
    input must be the unfolded input to the convolution (see unfold_func)
    and grad_output the backpropagated gradient
    """
    grad_output_viewed = separate_channels_and_pixels(module, grad_output)
    AX = einsum("nkl,vnml->vnkm", (input, grad_output_viewed))
    sum_dims = [0,1] if sum_batch else [0]
    transpose_dims = (0,1) if sum_batch else (1,2)
    weight_diagonal = (AX ** 2).sum(sum_dims).transpose(*transpose_dims)
    if sum_batch:
        return weight_diagonal.view_as(module.weight)
    else:
        return weight_diagonal.reshape(-1, *module.weight.numpy().shape)


def extract_bias_diagonal(module, sqrt, N, sum_batch=True):
    """
    `sqrt` must be the backpropagated quantity for DiagH or DiagGGN(MC)
    """
    V_axis, N_axis = 0, 1

    if N == 1:
        einsum_eq = "vncl->vnc"
    elif N == 2:
        einsum_eq = "vnchw->vnc"
    elif N == 3:
        einsum_eq = "vncdhw->vnc"
    else:
        ValueError("{}-dimensional Conv. is not implemented.".format(N))
    sum_dims = [V_axis, N_axis] if sum_batch else [V_axis]
    return (einsum(einsum_eq, sqrt) ** 2).sum(sum_dims)


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
