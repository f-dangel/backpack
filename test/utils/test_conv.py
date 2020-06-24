"""Test generalization of unfold to 3d convolutions."""

import torch
import pytest

from backpack.utils.conv import unfold_by_conv, unfold_func
from test.utils.test_conv_settings import SETTINGS
from ..automated_test import check_sizes_and_values
from test.core.derivatives.problem import make_test_problems


PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]

CONV2D_PROBLEMS = [
    problem
    for problem in PROBLEMS
    if isinstance(problem.make_module(), torch.nn.Conv2d)
]
CONV2D_IDS = [problem.make_id() for problem in CONV2D_PROBLEMS]


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


@pytest.mark.parametrize("problem", CONV2D_PROBLEMS, ids=CONV2D_IDS)
def test_unfold_by_conv(problem):
    """Test the Unfold by convolution for torch.nn.Conv2d.

    Args:
        problem (ConvProblem): Problem for testing unfold operation.
    """
    problem.set_up()
    input = torch.rand(problem.input_shape).to(problem.device)

    result_unfold = unfold_func(problem.module)(input)
    result_unfold_by_conv = unfold_by_conv(input, problem.module)

    check_sizes_and_values(result_unfold, result_unfold_by_conv)
    problem.tear_down()


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_convolution_with_unfold(problem):
    """Test the Unfold operation of torch.nn.Conv1d and torch.nn.Conv3d
    by using convolution.

    Args:
        problem (ConvProblem): Problem for testing unfold operation.
    """
    problem.set_up()
    input = torch.rand(problem.input_shape).to(problem.device)

    result_conv = problem.module(input)
    result_conv_by_unfold = convolution_with_unfold(input, problem.module)

    check_sizes_and_values(result_conv, result_conv_by_unfold)
    problem.tear_down()
