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
SETTINGS = [
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

for module, in_shape in SETTINGS:
    input = torch.rand(in_shape)

    result_unfold = unfold(input, module)
    result_unfold_by_conv = unfold_by_conv(input, module)

    if not torch.allclose(result_unfold, result_unfold):
        raise ValueError("[FAILED-unfold] {}".format(module))
    else:
        print("[PASSED] {}".format(module))


def convolution_with_unfold(input, module):
    """Perform convlution via matrix multiplication."""
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


SETTINGS += [
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
for module, in_shape in SETTINGS:
    input = torch.rand(in_shape)

    result_conv = module(input)
    result_conv_by_unfold = convolution_with_unfold(input, module)

    if not torch.allclose(result_conv, result_conv_by_unfold, rtol=1e-5, atol=1e-6):
        raise ValueError("[FAILED-conv] {}".format(module))
    else:
        print("[PASSED] {}".format(module))
