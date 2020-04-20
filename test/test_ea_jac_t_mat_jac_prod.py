"""Test KFRA backpropagation

H_in →  1/N ∑ₙ Jₙ^T H_out Jₙ

"""
import pytest
import torch
from torch.nn import (  # AvgPool2d,; Dropout,
    Conv2d,
    Linear,
    MaxPool2d,
    Sigmoid,
    Tanh,
    ZeroPad2d,
)

from backpack import extend
from backpack.core.derivatives import derivatives_for
from backpack.hessianfree.lop import transposed_jacobian_vector_product

from .automated_test import check_sizes, check_values


def get_output_shape(input, layer):
    return layer(input).shape


def autograd_ea_jac_t_mat_jac_prod(layer, input, mat):
    def sample_jac_t_mat_jac_prod(layer, sample, mat):
        assert sample.shape[0] == 1, "input is not batch size 1: {}".format(
            sample.shape
        )
        assert len(mat.shape) == 2

        def sample_jac_t_mat_prod(layer, sample, mat):
            result = torch.zeros(sample.numel(), mat.size(1))

            sample.requires_grad = True
            output = layer(sample)

            for col in range(mat.size(1)):
                column = mat[:, col].reshape(output.shape)
                result[:, col] = transposed_jacobian_vector_product(
                    [output], [sample], [column], retain_graph=True
                )[0].reshape(-1)

            return result

        jac_t_mat = sample_jac_t_mat_prod(layer, sample, mat)
        mat_t_jac = jac_t_mat.t()
        jac_t_mat_t_jac = sample_jac_t_mat_prod(layer, sample, mat_t_jac)
        jac_t_mat_jac = jac_t_mat_t_jac.t()

        return jac_t_mat_jac

    N = input.shape[0]
    input_features = input.shape.numel() // N

    result = torch.zeros(input_features, input_features)

    for n in range(N):
        sample_n = input[n].unsqueeze(0)
        result += sample_jac_t_mat_jac_prod(layer, sample_n, mat)

    return result / N


def derivative_from_layer(layer):
    layer_to_derivative = derivatives_for

    for module_cls, derivative_cls in layer_to_derivative.items():
        if isinstance(layer, module_cls):
            return derivative_cls()

    raise RuntimeError("No derivative available for {}".format(layer))


def backpack_ea_jac_t_mat_jac_prod(layer, input, mat):
    layer = extend(layer)
    derivative = derivative_from_layer(layer)

    # forward pass to initialize backpack buffers
    _ = layer(input)

    return derivative.ea_jac_t_mat_jac_prod(layer, None, None, mat)


def generate_data_ea_jac_t_mat_jac_prod(layer, input_shape):
    input = torch.rand(input_shape)
    out_features = get_output_shape(input, layer)[1:].numel()
    mat = torch.rand(out_features, out_features)
    return input, mat


def make_id(layer, input_shape):
    return "in{}-{}".format(input_shape, layer)


torch.manual_seed(0)
ARGS = "layer,input_shape"
SETTINGS = [
    # (layer, input_shape)
    [Linear(20, 10), (5, 20)],
    [MaxPool2d(kernel_size=2), (5, 3, 10, 8)],
    [MaxPool2d(kernel_size=2), (1, 2, 4, 4)],
    [MaxPool2d(kernel_size=3, stride=2, padding=1), (3, 2, 9, 11)],
    # [AvgPool2d(kernel_size=3), (3, 4, 7, 6)],
    [Sigmoid(), (6, 20)],
    [Sigmoid(), (6, 2, 7)],
    [Tanh(), (6, 20)],
    [Tanh(), (6, 2, 7)],
    [Conv2d(2, 3, kernel_size=2), (3, 2, 7, 9)],
    [Conv2d(2, 3, kernel_size=2, padding=1), (3, 2, 7, 9)],
    # [Conv2d(2, 3, kernel_size=2, padding=1, stride=2), (3, 2, 7, 9)],
    [Conv2d(2, 3, kernel_size=2, padding=1, stride=1, dilation=2), (3, 2, 7, 9)],
    [ZeroPad2d(2), (4, 3, 4, 5)],
    # not deterministic
    # [Dropout(0.2), (5, 10, 4)],
]
IDS = [make_id(layer, input_shape) for (layer, input_shape) in SETTINGS]


@pytest.mark.parametrize(ARGS, SETTINGS, ids=IDS)
def test_ea_jac_t_mat_jac_prod(layer, input_shape):
    torch.manual_seed(0)
    input, mat = generate_data_ea_jac_t_mat_jac_prod(layer, input_shape)
    return _compare_ea_jac_t_mat_jac_prod(layer, input, mat)


def _compare_ea_jac_t_mat_jac_prod(layer, input, mat):
    autograd_result = autograd_ea_jac_t_mat_jac_prod(layer, input, mat)
    backpack_result = backpack_ea_jac_t_mat_jac_prod(layer, input, mat)

    check_sizes(autograd_result, backpack_result)
    check_values(autograd_result, backpack_result)

    return backpack_result


def test_ea_jac_t_mat_jac_prod_linear_manual():
    # Linear with manual ea_jac_t_mat_jac_prod
    input_shape = (5, 13)
    layer = torch.nn.Linear(13, 10)

    input, mat = generate_data_ea_jac_t_mat_jac_prod(layer, input_shape)

    test_result = _compare_ea_jac_t_mat_jac_prod(layer, input, mat)
    manual_result = torch.einsum(
        "ab,ac,cd->bd", layer.weight.data, mat, layer.weight.data
    )

    check_sizes(test_result, manual_result)
    check_values(test_result, manual_result)
