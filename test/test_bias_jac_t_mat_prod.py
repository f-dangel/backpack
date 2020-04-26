"""Test transposed Jacobian-vector product w.r.t. layer bias.

Notes:
    - Only transposed Jacobian-vector products (V=1) are tested here.
"""
import pytest
import torch
from torch.nn import Conv2d, ConvTranspose2d, Linear

from backpack import extend
from backpack.hessianfree.lop import transposed_jacobian_vector_product

from .automated_test import check_sizes, check_values
from .test_ea_jac_t_mat_jac_prod import derivative_from_layer, get_output_shape


def make_id(layer, input_shape, sum_batch):
    return "in{}-{}-{}".format(input_shape, layer, sum_batch)


torch.manual_seed(0)
ARGS = "layer,input_shape,sum_batch"
SETTINGS = [
    # (layer, input_shape, sum_batch)
    # sum_batch = True
    [Linear(5, 1), (3, 5), True],
    [Linear(20, 10), (5, 20), True],
    [Conv2d(2, 3, kernel_size=2), (3, 2, 11, 13), True],
    [Conv2d(2, 3, kernel_size=2, padding=1), (3, 2, 11, 13), True],
    [Conv2d(2, 3, kernel_size=2, padding=1, stride=2), (3, 2, 11, 13), True],
    [
        Conv2d(2, 3, kernel_size=2, padding=1, stride=2, dilation=2),
        (3, 2, 11, 13),
        True,
    ],
    [ConvTranspose2d(2, 3, kernel_size=2), (3, 2, 11, 13), True],
    [ConvTranspose2d(2, 3, kernel_size=2, padding=1), (3, 2, 11, 13), True],
    [ConvTranspose2d(2, 3, kernel_size=2, padding=1, stride=2), (3, 2, 11, 13), True],
    [
        ConvTranspose2d(2, 3, kernel_size=2, padding=1, stride=2, dilation=2),
        (3, 2, 11, 13),
        True,
    ],
    # sum_batch = False
    [Linear(5, 1), (3, 5), False],
    [Linear(20, 10), (5, 20), False],
    [Conv2d(2, 3, kernel_size=2), (3, 2, 11, 13), False],
    [Conv2d(2, 3, kernel_size=2, padding=1), (3, 2, 11, 13), False],
    [Conv2d(2, 3, kernel_size=2, padding=1, stride=2), (3, 2, 11, 13), False],
    [
        Conv2d(2, 3, kernel_size=2, padding=1, stride=2, dilation=2),
        (3, 2, 11, 13),
        False,
    ],
    [ConvTranspose2d(2, 3, kernel_size=2), (3, 2, 11, 13), False],
    [ConvTranspose2d(2, 3, kernel_size=2, padding=1), (3, 2, 11, 13), False],
    [ConvTranspose2d(2, 3, kernel_size=2, padding=1, stride=2), (3, 2, 11, 13), False],
    [
        ConvTranspose2d(2, 3, kernel_size=2, padding=1, stride=2, dilation=2),
        (3, 2, 11, 13),
        False,
    ],
]
IDS = [
    make_id(layer, input_shape, sum_batch)
    for (layer, input_shape, sum_batch) in SETTINGS
]


def autograd_bias_jac_t_mat_prod(layer, input, mat, sum_batch):

    if sum_batch:
        output = layer(input)
        return transposed_jacobian_vector_product([output], [layer.bias], [mat])[0]
    else:
        N = input.shape[0]

        individual_jac_t_mat_prods = []
        for n in range(N):
            sample = input[n].unsqueeze(0)
            output = layer(sample)
            vec = mat[n].unsqueeze(0)
            individual_jac_t_mat_prods.append(
                transposed_jacobian_vector_product([output], [layer.bias], [vec])[
                    0
                ].unsqueeze(0)
            )

        return torch.cat(individual_jac_t_mat_prods)


def backpack_bias_jac_t_mat_prod(layer, input, mat, sum_batch):
    layer = extend(layer)
    derivative = derivative_from_layer(layer)

    # forward pass to initialize backpack buffers
    _ = layer(input)

    return derivative.bias_jac_t_mat_prod(layer, None, None, mat, sum_batch=sum_batch)


def generate_data_bias_jac_t_mat_prod(layer, input_shape):
    input = torch.rand(input_shape)
    output_shape = get_output_shape(input, layer)
    mat = torch.rand(output_shape)
    return input, mat


@pytest.mark.parametrize(ARGS, SETTINGS, ids=IDS)
def test_bias_jac_t_mat_prod(layer, input_shape, sum_batch):
    torch.manual_seed(0)
    input, mat = generate_data_bias_jac_t_mat_prod(layer, input_shape)
    return _compare_bias_jac_t_mat_prod(layer, input, mat, sum_batch)


def _compare_bias_jac_t_mat_prod(layer, input, mat, sum_batch):
    autograd_result = autograd_bias_jac_t_mat_prod(layer, input, mat, sum_batch)
    backpack_result = backpack_bias_jac_t_mat_prod(layer, input, mat, sum_batch)

    print(autograd_result.shape)
    print(backpack_result.shape)

    check_sizes(autograd_result, backpack_result)
    check_values(autograd_result, backpack_result)

    return backpack_result
