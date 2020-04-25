"""Generalize unfold to 3d convolutions and transpose convolutions.

Background:
    (Transposed) Convolutions can be implemented as matrix multiplications.
    The multiplication involves (i) a matrix view of the convolution kernel
    and (ii) a matrix storing the patches (the data that the kernel is con-
    tracted with during one sliding iteration of a convolution).

    Because of this matrix, the L2 norm and SGS in BackPACK are much more
    memory efficient than a naive implementation via individual gradients.

    It is also used to compute a Kronecker factor in KFAC/KFRA/KFLR.

    However, we can only get access to the latter 'unfolded input' matrix
    for 1d, 2d convolutions.

Goal:
    1) Is it possible to get the unfolded input matrix via convolutions
       with kernels with all elements zero except one? How efficient is that?
    2) Can we generalize this to transposed convolutions?

"""

import torch

###############################################################################
#      Get the unfolded input with a convolution instead of torch.unfold      #
###############################################################################


def unfold(input, module):
    """Return unfolded input."""
    return torch.nn.Unfold(
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )(input)


def unfold_by_conv(input, module):
    """Return the unfolded input using convolution"""
    N, C_in, H_in, W_in = input.shape
    K_H, K_W = module.kernel_size

    def make_weight():
        weight = torch.zeros(K_H * K_W, 1, K_H, K_W)

        for i in range(K_H * K_W):
            extraction = torch.zeros(K_H * K_W)
            extraction[i] = 1.0
            weight[i] = extraction.reshape(1, K_H, K_W)

        return weight.repeat(C_in, 1, 1, 1)

    # might change for groups !=1
    groups = C_in

    unfold_module = torch.nn.Conv2d(
        in_channels=C_in,
        out_channels=C_in * K_H * K_W,
        kernel_size=module.kernel_size,
        stride=module.stride,
        padding=module.padding,
        dilation=module.dilation,
        bias=False,
        groups=groups,
    )

    unfold_weight = make_weight()
    if unfold_weight.shape != unfold_module.weight.shape:
        raise ValueError(
            "weight should have shape {}, got {}".format(
                unfold_module.weight.shape, unfold_weight.shape
            )
        )

    unfold_module.weight.data = unfold_weight

    unfold_output = unfold_module(input)
    unfold_output = unfold_output.reshape(N, -1, K_H * K_W)

    return unfold_output


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

print("Unfold check")
for module, in_shape in UNFOLD_SETTINGS:
    input = torch.rand(in_shape)

    result_unfold = unfold(input, module)
    result_unfold_by_conv = unfold_by_conv(input, module)

    if not torch.allclose(result_unfold, result_unfold):
        raise ValueError("[FAILED-unfold] {}".format(module))
    else:
        print("[PASSED] {}".format(module))

###############################################################################
#             Perform a convolution with the unfolded input matrix            #
###############################################################################


def convolution_with_unfold(input, module):
    """Perform convlution via matrix multiplication."""
    assert module.bias is None

    N, C_in, H_in, W_in = input.shape
    _, _, H_out, W_out = get_output_shape(input, module)
    C_out, _, K_H, K_W = module.weight.shape

    G = module.groups

    weight_matrix = module.weight.data.reshape(C_out, C_in // G, K_H, K_W)
    weight_matrix = module.weight.data.reshape(G, C_out // G, C_in // G, K_H, K_W)
    unfolded_input = unfold_by_conv(input, module).reshape(
        N, G, C_in // G, K_H, K_W, H_out, W_out
    )
    result = torch.einsum("gocxy,ngcxyhw->ngohw", weight_matrix, unfolded_input)

    result = result.reshape(N, C_out, H_out, W_out)
    return result


def get_output_shape(input, module):
    return module(input).shape


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

print("Conv via unfold check")
for module, in_shape in CONV_SETTINGS:
    input = torch.rand(in_shape)

    result_conv = module(input)
    result_conv_by_unfold = convolution_with_unfold(input, module)

    if not torch.allclose(result_conv, result_conv_by_unfold, rtol=1e-5, atol=1e-6):
        raise ValueError("[FAILED-conv] {}".format(module))
    else:
        print("[PASSED] {}".format(module))

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
    ## no bias
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
    ## with bias
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

###############################################################################
#                  Generalize unfold to input data for Conv3d                 #
###############################################################################
