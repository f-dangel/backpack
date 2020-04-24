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

"""Example 1: one sample, C_in = 1, C_out = 1"""
N = 1
C_in = 1
sample = torch.Tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]]).float()
print(sample.shape)

C_out = 1
K_H = 2
K_W = 2
kernel = torch.Tensor([[[[1, -1], [2, 3]]]]).float()
# print(kernel.shape)

layer = torch.nn.Conv2d(1, 1, kernel_size=2, bias=False)
layer.weight.data = kernel
# print(layer)


output = layer(sample)
# print(output.data)

unfolded_sample = torch.nn.Unfold(kernel_size=layer.kernel_size,)(sample)
print(unfolded_sample.shape)

# Let's try to do that unfold with convolutions

groups = C_in
unfold_kernel = torch.zeros(K_H * K_W, 1, K_H, K_W)

for i in range(K_H * K_W):
    extraction = torch.zeros(K_H * K_W)
    extraction[i] = 1.0
    extraction = extraction.reshape(1, K_H, K_W)

    unfold_kernel[i] = extraction

# print(unfold_kernel)
# print(unfold_kernel.shape)

unfold_module = layer = torch.nn.Conv2d(
    C_in, K_H * K_W, kernel_size=layer.kernel_size, bias=False, groups=C_in
)

# print(unfold_module.weight.shape)
unfold_module.weight.data = unfold_kernel

unfold_output = unfold_module(sample)
# print(unfold_output)
# print(unfold_output.shape)
# print(unfolded_sample)
# print(unfolded_sample.shape)

unfold_output = unfold_output.reshape(N, -1, K_H * K_W)

print("Unfold via torch.nn.Conv2d (idea)")
print(unfold_output)
print(unfold_output.shape)
print("Unfold via torch.nn.Unfold:")
print(unfolded_sample)
print(unfolded_sample.shape)

print("Match: {}".format(torch.allclose(unfold_output, unfolded_sample)))

# Let's do this with a function


def unfold(input, module):
    """Return unfolded input."""
    if len(input.shape) != 4:
        raise NotImplementedError("Only 4d-input supported, got {}".format(input.shape))

    return torch.nn.Unfold(
        kernel_size=module.kernel_size,
        dilation=module.dilation,
        padding=module.padding,
        stride=module.stride,
    )(input)


def unfold_by_conv(input, module):
    """Return the unfolded input using convolution"""
    assert module.groups == 1
    assert len(input.shape) == 4

    N, C_in, H_in, W_in = input.shape
    K_H, K_W = module.kernel_size

    def make_weight():
        weight = torch.zeros(K_H * K_W, 1, K_H, K_W)

        for i in range(K_H * K_W):
            extraction = torch.zeros(K_H * K_W)
            extraction[i] = 1.0
            weight[i] = extraction.reshape(1, K_H, K_W)

        return weight

    # print(weight)
    # print(weight.shape)

    # might change for groups !=1
    groups = C_in

    unfold_module = torch.nn.Conv2d(
        in_channels=C_in,
        out_channels=K_H * K_W,
        kernel_size=layer.kernel_size,
        stride=layer.stride,
        dilation=layer.dilation,
        bias=False,
        groups=groups,
    )

    unfold_weight = make_weight()
    assert unfold_weight.shape == unfold_module.weight.shape

    # print(unfold_module.weight.shape)
    unfold_module.weight.data = unfold_weight

    unfold_output = unfold_module(sample)
    # print(unfold_output)
    # print(unfold_output.shape)
    # print(unfolded_sample)
    # print(unfolded_sample.shape)

    unfold_output = unfold_output.reshape(N, -1, K_H * K_W)

    return unfold_output


# check that
SETTINGS = [
    [torch.nn.Conv2d(1, 1, kernel_size=2, bias=False), (1, 1, 3, 3)],
    [torch.nn.Conv2d(2, 3, kernel_size=2), (3, 2, 11, 13)],
    # [torch.nn.Conv2d(2, 3, kernel_size=2, padding=1), (3, 2, 11, 13)],
    # [torch.nn.Conv2d(2, 3, kernel_size=2, padding=1, stride=2), (3, 2, 11, 13)],
    # [
    #     torch.nn.Conv2d(2, 3, kernel_size=2, padding=1, stride=2, dilation=2),
    #     (3, 2, 11, 13),
    # ],
]

for module, in_shape in SETTINGS:
    input = torch.rand(in_shape)

    result_unfold = unfold(input, module)
    result_unfold_by_conv = unfold_by_conv(input, module)

    if not torch.allclose(result_unfold, result_unfold):
        print("[FAILED] {}".format(module))
        print(result_unfold)
        print(result_unfold_by_conv)
    else:
        print("[PASSED] {}".format(module))
