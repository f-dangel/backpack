import torch
from torch import einsum
from torch.nn.functional import conv_transpose1d, conv_transpose2d, conv_transpose3d

from einops import rearrange
from backpack.utils.conv import separate_channels_and_pixels


def get_weight_gradient_factors(input, grad_out, module, N):
    M, C_in = input.shape[0], input.shape[1]
    kernel_size = module.kernel_size
    kernel_size_numel = int(torch.prod(torch.Tensor(kernel_size)))

    X = unfold_by_conv_transpose(input, module).reshape(M, C_in * kernel_size_numel, -1)
    dE_dY = rearrange(grad_out, "n c ... -> n c (...)")

    return X, dE_dY


def extract_weight_diagonal(module, input, grad_output, N, sum_batch=True):
    """
    input must be the unfolded input to the convolution (see unfold_func)
    and grad_output the backpropagated gradient
    """
    out_channels = module.weight.shape[0]
    in_channels = module.weight.shape[1]
    k = module.weight.shape[2:]

    M = input.shape[0]
    output_shape = module(module.input0).shape
    spatial_out_size = output_shape[2:]
    spatial_out_numel = spatial_out_size.numel()

    input_reshaped = input.reshape(M, -1, spatial_out_numel)

    V_axis, N_axis = 0, 1
    grad_output_viewed = separate_channels_and_pixels(module, grad_output)
    AX = einsum("nkl,vnml->vnkm", (input_reshaped, grad_output_viewed))
    N = AX.shape[N_axis]
    sum_dims = [V_axis, N_axis] if sum_batch else [V_axis]
    transpose_dims = (V_axis, N_axis) if sum_batch else (V_axis + 1, N_axis + 1)
    weight_diagonal = (AX ** 2).sum(sum_dims).transpose(*transpose_dims)
    if sum_batch:
        weight_diagonal = weight_diagonal.reshape(in_channels, out_channels, *k)
    else:
        weight_diagonal = weight_diagonal.reshape(N, in_channels, out_channels, *k)
    return weight_diagonal.transpose(*transpose_dims)


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
        ValueError("{}-dimensional ConvTranspose is not implemented.".format(N))
    sum_dims = [V_axis, N_axis] if sum_batch else [V_axis]
    return (einsum(einsum_eq, sqrt) ** 2).sum(sum_dims)


def unfold_by_conv_transpose(input, module):
    """Return the unfolded input using transpose convolution."""
    N, C_in = input.shape[0], input.shape[1]
    kernel_size = module.kernel_size
    kernel_size_numel = int(torch.prod(torch.Tensor(kernel_size)))

    def make_weight():
        weight = torch.zeros(1, kernel_size_numel, *kernel_size)

        for i in range(kernel_size_numel):
            extraction = torch.zeros(kernel_size_numel)
            extraction[i] = 1.0
            weight[0, i] = extraction.reshape(*kernel_size)

        repeat = [C_in, 1] + [1 for _ in kernel_size]
        weight = weight.repeat(*repeat)
        return weight.to(module.weight.device)

    def get_conv_transpose():
        functional_for_module_cls = {
            torch.nn.ConvTranspose1d: conv_transpose1d,
            torch.nn.ConvTranspose2d: conv_transpose2d,
            torch.nn.ConvTranspose3d: conv_transpose3d,
        }
        return functional_for_module_cls[module.__class__]

    conv_transpose = get_conv_transpose()
    unfold = conv_transpose(
        input,
        make_weight().to(module.weight.device),
        bias=None,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        groups=C_in,
    )

    return unfold.reshape(N, -1, kernel_size_numel)
