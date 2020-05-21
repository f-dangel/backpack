"""Test generalization of unfold to 3d convolutions."""

# TODO: @sbharadwajj: impose test suite structure

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


###############################################################################
#                         Embed Conv(N-1)d in Conv(N)d                        #
###############################################################################


def add_dimension(input, module):
    """Transform input and module of `Conv(N-1)d` into `Conv(N)d`."""
    new_input = add_input_dimension(input)
    new_module = add_module_dimension(module)

    return new_input, new_module


def remove_dimension(input, module):
    """Transform input and module of `Conv(N)d` into `Conv(N-1)d`."""
    new_input = remove_input_dimension(input)
    new_module = remove_module_dimension(module)

    return new_input, new_module


def add_input_dimension(input):
    new_input = input.unsqueeze(-1)
    return new_input


def remove_input_dimension(input):
    new_input = input.squeeze(-1)
    return new_input


def add_module_dimension(module):
    if isinstance(module, torch.nn.Conv1d):
        new_module_cls = torch.nn.Conv2d
    elif isinstance(module, torch.nn.Conv2d):
        new_module_cls = torch.nn.Conv3d
    else:
        raise ValueError("Cannot add dimension to {}".format(module))

    new_out_channels = module.out_channels
    new_in_channels = module.in_channels
    new_kernel_size = module.kernel_size + (1,)
    new_stride = module.stride + (1,)
    new_padding = module.padding + (0,)
    new_dilation = module.dilation + (1,)
    new_groups = module.groups
    new_bias = module.bias is not None
    new_padding_mode = module.padding_mode

    new_module = new_module_cls(
        out_channels=new_out_channels,
        in_channels=new_in_channels,
        kernel_size=new_kernel_size,
        stride=new_stride,
        padding=new_padding,
        dilation=new_dilation,
        groups=new_groups,
        bias=new_bias,
        padding_mode=new_padding_mode,
    )

    new_weight_data = module.weight.data.unsqueeze(-1)
    new_module.weight.data = new_weight_data

    if new_bias:
        new_bias_data = module.bias.data
        new_module.bias.data = new_bias_data

    return new_module


def remove_module_dimension(module):
    if isinstance(module, torch.nn.Conv3d):
        new_module_cls = torch.nn.Conv2d
    elif isinstance(module, torch.nn.Conv2d):
        new_module_cls = torch.nn.Conv1d
    else:
        raise ValueError("Cannot remove dimension of {}".format(module))

    new_out_channels = module.out_channels
    new_in_channels = module.in_channels

    assert module.kernel_size[-1] == 1
    new_kernel_size = module.kernel_size[:-1]

    assert module.stride[-1] == 1
    new_stride = module.stride[:-1]

    assert module.padding[-1] == 0
    new_padding = module.padding[:-1]

    assert module.dilation[-1] == 1
    new_dilation = module.dilation[:-1]

    new_groups = module.groups
    new_bias = module.bias is not None
    new_padding_mode = module.padding_mode

    new_module = new_module_cls(
        out_channels=new_out_channels,
        in_channels=new_in_channels,
        kernel_size=new_kernel_size,
        stride=new_stride,
        padding=new_padding,
        dilation=new_dilation,
        groups=new_groups,
        bias=new_bias,
        padding_mode=new_padding_mode,
    )

    assert module.weight.shape[-1] == 1
    new_weight_data = module.weight.data.squeeze(-1)
    new_module.weight.data = new_weight_data

    if new_bias:
        new_bias_data = module.bias.data
        new_module.bias.data = new_bias_data

    return new_module


INCREASE_SETTINGS = CONV_SETTINGS + [
    # 1d convolution
    # no bias
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
    # with bias
    [torch.nn.Conv1d(1, 1, kernel_size=2, bias=True), (1, 1, 3)],
    [torch.nn.Conv1d(1, 2, kernel_size=2, bias=True), (1, 1, 3)],
    [torch.nn.Conv1d(2, 1, kernel_size=2, bias=True), (1, 2, 3)],
    [torch.nn.Conv1d(2, 2, kernel_size=2, bias=True), (1, 2, 3)],
    [torch.nn.Conv1d(2, 3, kernel_size=2, bias=True), (3, 2, 11)],
    [torch.nn.Conv1d(2, 3, kernel_size=2, padding=1, bias=True), (3, 2, 11)],
    [torch.nn.Conv1d(2, 3, kernel_size=2, padding=1, stride=2, bias=True), (3, 2, 11),],
    [
        torch.nn.Conv1d(
            2, 3, kernel_size=2, padding=1, stride=2, dilation=2, bias=True
        ),
        (3, 2, 11),
    ],
    # 2d convolution with bias
    [torch.nn.Conv2d(1, 1, kernel_size=2, bias=True), (1, 1, 3, 3)],
    [torch.nn.Conv2d(1, 2, kernel_size=2, bias=True), (1, 1, 3, 3)],
    [torch.nn.Conv2d(2, 1, kernel_size=2, bias=True), (1, 2, 3, 3)],
    [torch.nn.Conv2d(2, 2, kernel_size=2, bias=True), (1, 2, 3, 3)],
    [torch.nn.Conv2d(2, 3, kernel_size=2, bias=True), (3, 2, 11, 13)],
    [torch.nn.Conv2d(2, 3, kernel_size=2, padding=1, bias=True), (3, 2, 11, 13)],
    [
        torch.nn.Conv2d(2, 3, kernel_size=2, padding=1, stride=2, bias=True),
        (3, 2, 11, 13),
    ],
    [
        torch.nn.Conv2d(
            2, 3, kernel_size=2, padding=1, stride=2, dilation=2, bias=True
        ),
        (3, 2, 11, 13),
    ],
]


def test_conv1d2d_add_remove_dimension():
    print("Conv embedding in higher dimension")
    for module, in_shape in INCREASE_SETTINGS:
        print("[TRY] {}".format(module))
        input = torch.rand(in_shape)

        result_conv = module(input)

        # in higher dim
        new_input, new_module = add_dimension(input, module)
        result_conv2 = remove_input_dimension(new_module(new_input))

        # if not torch.allclose(result_conv, result_conv2, atol=1e-6):
        # for res1, res2 in zip(result_conv.flatten(), result_conv2.flatten()):
        # print(res1, "should be", res2, "ratio:", res1 / res2)
        assert torch.allclose(result_conv, result_conv2, atol=1e-6)

        # adding and removing a dimension are inverses
        old_input, old_module = remove_dimension(new_input, new_module)
        assert torch.allclose(old_input, input)
        result_conv3 = old_module(old_input)

        assert torch.allclose(result_conv, result_conv3)

        print("[PASSED] {}".format(module))
