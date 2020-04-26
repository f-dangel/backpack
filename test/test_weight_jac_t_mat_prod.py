"""Test transposed Jacobian-vector product w.r.t. layer weight.

Notes:
    - Only transposed Jacobian-vector products (V=1) are tested here.
"""

import pytest
import torch

from backpack import extend
from backpack.hessianfree.lop import transposed_jacobian_vector_product

from .automated_test import check_sizes, check_values
from .test_bias_jac_t_mat_prod import ARGS, SETTINGS, make_id
from .test_ea_jac_t_mat_jac_prod import derivative_from_layer, get_output_shape

IDS = [
    make_id(layer, input_shape, sum_batch)
    for (layer, input_shape, sum_batch) in SETTINGS
]


def autograd_weight_jac_t_mat_prod(layer, input, mat, sum_batch):

    if sum_batch:
        output = layer(input)
        return transposed_jacobian_vector_product([output], [layer.weight], [mat])[0]
    else:
        N = input.shape[0]

        individual_jac_t_mat_prods = []
        for n in range(N):
            sample = input[n].unsqueeze(0)
            output = layer(sample)
            vec = mat[n].unsqueeze(0)
            individual_jac_t_mat_prods.append(
                transposed_jacobian_vector_product([output], [layer.weight], [vec])[
                    0
                ].unsqueeze(0)
            )

        return torch.cat(individual_jac_t_mat_prods)


def backpack_weight_jac_t_mat_prod(layer, input, mat, sum_batch):
    layer = extend(layer)
    derivative = derivative_from_layer(layer)

    # forward pass to initialize backpack buffers
    _ = layer(input)

    return derivative.weight_jac_t_mat_prod(layer, None, None, mat, sum_batch=sum_batch)


def generate_data_weight_jac_t_mat_prod(layer, input_shape):
    input = torch.rand(input_shape)
    output_shape = get_output_shape(input, layer)
    mat = torch.rand(output_shape)
    return input, mat


@pytest.mark.parametrize(ARGS, SETTINGS, ids=IDS)
def test_weight_jac_t_mat_prod(layer, input_shape, sum_batch):
    torch.manual_seed(0)
    input, mat = generate_data_weight_jac_t_mat_prod(layer, input_shape)
    return _compare_weight_jac_t_mat_prod(layer, input, mat, sum_batch)


def _compare_weight_jac_t_mat_prod(layer, input, mat, sum_batch):
    autograd_result = autograd_weight_jac_t_mat_prod(layer, input, mat, sum_batch)
    backpack_result = backpack_weight_jac_t_mat_prod(layer, input, mat, sum_batch)

    print(autograd_result.shape)
    print(backpack_result.shape)

    check_sizes(autograd_result, backpack_result)
    check_values(autograd_result, backpack_result)

    return backpack_result
