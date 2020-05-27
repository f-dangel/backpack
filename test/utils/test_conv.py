"""Test generalization of unfold to 3d convolutions."""

# TODO: @sbharadwajj: impose test suite structure
# TODO: @sbharadwajj: Test for groups ≠ 1

import torch

from backpack.utils.conv import unfold_by_conv, unfold_func

from ..automated_test import check_sizes_and_values

###############################################################################
#      Get the unfolded input with a convolution instead of torch.unfold      #
###############################################################################

torch.manual_seed(0)

# check
UNFOLD_SETTINGS = [
    [torch.nn.Conv2d(1, 1, kernel_size=2, bias=False), (1, 1, 3, 3)],
    [torch.nn.Conv2d(1, 2, kernel_size=2, bias=False), (1, 1, 3, 3)],
    [torch.nn.Conv2d(2, 1, kernel_size=2, bias=False), (1, 2, 3, 3)],
    [torch.nn.Conv2d(2, 2, kernel_size=2, bias=False), (1, 2, 3, 3)],
    [torch.nn.Conv2d(2, 3, kernel_size=2, bias=False), (3, 2, 11, 13)],
    [torch.nn.Conv2d(2, 3, kernel_size=2, padding=1, bias=False), (3, 2, 11, 13)],
    [
        torch.nn.Conv2d(2, 3, kernel_size=2, padding=1, stride=2, bias=False),
        (3, 2, 11, 13),
    ],
    [
        torch.nn.Conv2d(
            2, 3, kernel_size=2, padding=1, stride=2, dilation=2, bias=False
        ),
        (3, 2, 11, 13),
    ],
]


def test_unfold_by_conv():
    for module, in_shape in UNFOLD_SETTINGS:
        input = torch.rand(in_shape)

        result_unfold = unfold_func(module)(input).flatten()
        result_unfold_by_conv = unfold_by_conv(input, module).flatten()

        check_sizes_and_values(result_unfold, result_unfold_by_conv)


###############################################################################
#             Perform a convolution with the unfolded input matrix            #
###############################################################################


def convolution_with_unfold(input, module):
    """Perform convolution via matrix multiplication."""
    assert module.bias is None

    def get_output_shape(input, module):
        return module(input).shape

    N, C_in = input.shape[0], input.shape[1]

    output_shape = get_output_shape(input, module)
    C_out = output_shape[1]
    spatial_out_size = output_shape[2:]
    spatial_out_numel = spatial_out_size.numel()

    kernel_size = module.kernel_size
    kernel_size_numel = int(torch.prod(torch.Tensor(kernel_size)))

    G = module.groups

    weight_matrix = module.weight.data.reshape(
        G, C_out // G, C_in // G, kernel_size_numel
    )
    unfolded_input = unfold_by_conv(input, module).reshape(
        N, G, C_in // G, kernel_size_numel, spatial_out_numel
    )

    result = torch.einsum("gocx,ngcxh->ngoh", weight_matrix, unfolded_input)

    return result.reshape(N, C_out, *spatial_out_size)


CONV_SETTINGS = UNFOLD_SETTINGS + [
    [
        torch.nn.Conv2d(
            2, 6, kernel_size=2, padding=1, stride=2, dilation=2, bias=False, groups=2
        ),
        (3, 2, 11, 13),
    ],
    [
        torch.nn.Conv2d(
            3, 6, kernel_size=2, padding=1, stride=2, dilation=2, bias=False, groups=3
        ),
        (5, 3, 11, 13),
    ],
    [
        torch.nn.Conv2d(
            16,
            33,
            kernel_size=(3, 5),
            stride=(2, 1),
            padding=(4, 2),
            dilation=(3, 1),
            bias=False,
        ),
        (20, 16, 50, 100),
    ],
]


def test_convolution2d_with_unfold():
    for module, in_shape in CONV_SETTINGS:
        input = torch.rand(in_shape)

        result_conv = module(input)
        result_conv_by_unfold = convolution_with_unfold(input, module)

        check_sizes_and_values(result_conv, result_conv_by_unfold)


CONV_1D_SETTINGS = [
    [torch.nn.Conv1d(1, 1, kernel_size=2, bias=False), (1, 1, 3)],
    [torch.nn.Conv1d(1, 2, kernel_size=2, bias=False), (1, 1, 3)],
    [torch.nn.Conv1d(2, 1, kernel_size=2, bias=False), (1, 2, 3)],
    [torch.nn.Conv1d(2, 2, kernel_size=2, bias=False), (1, 2, 3)],
    [torch.nn.Conv1d(2, 3, kernel_size=2, bias=False), (3, 2, 11)],
    [torch.nn.Conv1d(2, 3, kernel_size=2, padding=1, bias=False), (3, 2, 11)],
    [
        torch.nn.Conv1d(2, 3, kernel_size=2, padding=1, stride=2, bias=False),
        (3, 2, 11),
    ],
    [
        torch.nn.Conv1d(
            2, 3, kernel_size=2, padding=1, stride=2, dilation=2, bias=False
        ),
        (3, 2, 11),
    ],
]


def test_convolution1d_with_unfold():
    for module, in_shape in CONV_1D_SETTINGS:
        input = torch.rand(in_shape)

        result_conv = module(input)
        result_conv_by_unfold = convolution_with_unfold(input, module)

        check_sizes_and_values(result_conv, result_conv_by_unfold)


CONV_3D_SETTINGS = [
    [torch.nn.Conv3d(1, 1, kernel_size=2, bias=False), (1, 1, 3, 3, 3)],
    [torch.nn.Conv3d(1, 2, kernel_size=2, bias=False), (1, 1, 3, 3, 3)],
    [torch.nn.Conv3d(2, 1, kernel_size=2, bias=False), (1, 2, 3, 3, 3)],
    [torch.nn.Conv3d(2, 2, kernel_size=2, bias=False), (1, 2, 3, 3, 3)],
    [torch.nn.Conv3d(2, 3, kernel_size=2, bias=False), (3, 2, 11, 13, 17)],
    [torch.nn.Conv3d(2, 3, kernel_size=2, padding=1, bias=False), (3, 2, 11, 13, 17)],
    [
        torch.nn.Conv3d(2, 3, kernel_size=2, padding=1, stride=2, bias=False),
        (3, 2, 11, 13, 17),
    ],
    [
        torch.nn.Conv3d(
            2, 3, kernel_size=2, padding=1, stride=2, dilation=2, bias=False
        ),
        (3, 2, 11, 13, 17),
    ],
]


def test_convolution3d_with_unfold():
    print("Conv via unfold check")
    for module, in_shape in CONV_3D_SETTINGS:
        input = torch.rand(in_shape)

        result_conv = module(input)
        result_conv_by_unfold = convolution_with_unfold(input, module)

        check_sizes_and_values(result_conv, result_conv_by_unfold)
