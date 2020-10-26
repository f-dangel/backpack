"""Test generalization of unfold to transpose convolutions."""

import torch
import pytest

from backpack.utils.conv_transpose import unfold_by_conv_transpose
from test.utils.test_conv_transpose_settings import SETTINGS
from ..automated_test import check_sizes_and_values
from test.core.derivatives.problem import make_test_problems


PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


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


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_conv_transpose_with_unfold(problem):
    """Test the torch.nn.ConvTranspose() using the unfold operation
    and `conv_transpose_unfold` function.
    Args:
        problem (ConvProblem): Problem for testing torch.nn.ConvTranspose() operation.
    """
    problem.set_up()
    input = torch.rand(problem.input_shape).to(problem.device)

    result_conv_transpose = problem.module(input)
    result_conv_transpose_by_unfold = conv_transpose_with_unfold(input, problem.module)

    check_sizes_and_values(result_conv_transpose, result_conv_transpose_by_unfold)
    problem.tear_down()
