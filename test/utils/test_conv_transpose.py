"""Test generalization of unfold to transpose convolutions."""

# TODO: @sbharadwajj: impose test suite structure
# TODO: @sbharadwajj: test with groups≠1

import torch

from backpack.utils.conv_transpose import unfold_by_conv_transpose

from ..automated_test import check_sizes_and_values

torch.manual_seed(0)

###############################################################################
#             Perform a convolution with the unfolded input matrix            #
###############################################################################


def conv_transpose_with_unfold(input, module):
    """Perform transpose convolution via matrix multiplication."""
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
        C_in // G, G, C_out // G, kernel_size_numel
    )
    unfolded_input = unfold_by_conv_transpose(input, module).reshape(
        N, C_in // G, G, kernel_size_numel, spatial_out_numel
    )

    result = torch.einsum("cgox,ncgxh->ngoh", weight_matrix, unfolded_input)

    return result.reshape(N, C_out, *spatial_out_size)


CONV_TRANSPOSE_2D_SETTINGS = [
    [torch.nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False), (1, 1, 3, 3)],
    [torch.nn.ConvTranspose2d(1, 2, kernel_size=2, bias=False), (1, 1, 3, 3)],
    [torch.nn.ConvTranspose2d(2, 1, kernel_size=2, bias=False), (1, 2, 3, 3)],
    [torch.nn.ConvTranspose2d(2, 2, kernel_size=2, bias=False), (1, 2, 3, 3)],
    [torch.nn.ConvTranspose2d(2, 3, kernel_size=2, bias=False), (3, 2, 11, 13)],
    [
        torch.nn.ConvTranspose2d(2, 3, kernel_size=2, padding=1, bias=False),
        (3, 2, 11, 13),
    ],
    [
        torch.nn.ConvTranspose2d(2, 3, kernel_size=2, padding=1, stride=2, bias=False),
        (3, 2, 11, 13),
    ],
    [
        torch.nn.ConvTranspose2d(
            2, 3, kernel_size=2, padding=1, stride=2, dilation=2, bias=False
        ),
        (3, 2, 11, 13),
    ],
]


def test_conv_transpose2d_with_unfold():
    for module, in_shape in CONV_TRANSPOSE_2D_SETTINGS:
        input = torch.rand(in_shape)

        result_conv_transpose = module(input)
        result_conv_transpose_by_unfold = conv_transpose_with_unfold(input, module)

        check_sizes_and_values(result_conv_transpose, result_conv_transpose_by_unfold)


CONV_TRANSPOSE_1D_SETTINGS = [
    [torch.nn.ConvTranspose1d(1, 1, kernel_size=2, bias=False), (1, 1, 3)],
    [torch.nn.ConvTranspose1d(1, 2, kernel_size=2, bias=False), (1, 1, 3)],
    [torch.nn.ConvTranspose1d(2, 1, kernel_size=2, bias=False), (1, 2, 3)],
    [torch.nn.ConvTranspose1d(2, 2, kernel_size=2, bias=False), (1, 2, 3)],
    [torch.nn.ConvTranspose1d(2, 3, kernel_size=2, bias=False), (3, 2, 11)],
    [torch.nn.ConvTranspose1d(2, 3, kernel_size=2, padding=1, bias=False), (3, 2, 11)],
    [
        torch.nn.ConvTranspose1d(2, 3, kernel_size=2, padding=1, stride=2, bias=False),
        (3, 2, 11),
    ],
    [
        torch.nn.ConvTranspose1d(
            2, 3, kernel_size=2, padding=1, stride=2, dilation=2, bias=False
        ),
        (3, 2, 11),
    ],
]


def test_conv_transpose1d_with_unfold():
    for module, in_shape in CONV_TRANSPOSE_1D_SETTINGS:
        input = torch.rand(in_shape)

        result_conv_transpose = module(input)
        result_conv_transpose_by_unfold = conv_transpose_with_unfold(input, module)

        check_sizes_and_values(result_conv_transpose, result_conv_transpose_by_unfold)


CONV_TRANSPOSE_3D_SETTINGS = [
    [torch.nn.ConvTranspose3d(1, 1, kernel_size=2, bias=False), (1, 1, 3, 3, 3)],
    [torch.nn.ConvTranspose3d(1, 2, kernel_size=2, bias=False), (1, 1, 3, 3, 3)],
    [torch.nn.ConvTranspose3d(2, 1, kernel_size=2, bias=False), (1, 2, 3, 3, 3)],
    [torch.nn.ConvTranspose3d(2, 2, kernel_size=2, bias=False), (1, 2, 3, 3, 3)],
    [torch.nn.ConvTranspose3d(2, 3, kernel_size=2, bias=False), (3, 2, 11, 13, 17)],
    [
        torch.nn.ConvTranspose3d(2, 3, kernel_size=2, padding=1, bias=False),
        (3, 2, 11, 13, 17),
    ],
    [
        torch.nn.ConvTranspose3d(2, 3, kernel_size=2, padding=1, stride=2, bias=False),
        (3, 2, 11, 13, 17),
    ],
    [
        torch.nn.ConvTranspose3d(
            2, 3, kernel_size=2, padding=1, stride=2, dilation=2, bias=False
        ),
        (3, 2, 11, 13, 17),
    ],
]


def test_conv_transpose3d_with_unfold():
    for module, in_shape in CONV_TRANSPOSE_3D_SETTINGS:
        input = torch.rand(in_shape)

        result_conv_transpose = module(input)
        result_conv_transpose_by_unfold = conv_transpose_with_unfold(input, module)

        check_sizes_and_values(result_conv_transpose, result_conv_transpose_by_unfold)
