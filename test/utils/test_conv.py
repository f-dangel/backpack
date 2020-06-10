"""Test generalization of unfold to 3d convolutions."""

import torch
import pytest

from backpack.utils.conv import unfold_by_conv, unfold_func, convolution_with_unfold
from test.utils.test_conv_settings import SETTINGS
from ..automated_test import check_sizes_and_values
from test.core.derivatives.problem import make_test_problems


PROBLEMS = make_test_problems(SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]

Conv2d_PROBLEMS = [
    problem for problem in PROBLEMS if problem.module_fn.__name__ == "Conv2d"
]
Conv2d_IDS = [problem.make_id() for problem in Conv2d_PROBLEMS]


@pytest.mark.parametrize("problem", Conv2d_PROBLEMS, ids=Conv2d_IDS)
def test_unfold_by_conv(problem):
    """Test the Unfold by convolution for torch.nn.Conv2d.

    Args:
        problem (ConvProblem): Problem for testing unfold operation.
    """
    problem.set_up()
    input = torch.rand(problem.input_shape)

    result_unfold = unfold_func(problem.module)(input)
    result_unfold_by_conv = unfold_by_conv(input, problem.module)

    check_sizes_and_values(result_unfold, result_unfold_by_conv)


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_convolution_with_unfold(problem):
    """Test the Unfold operation of torch.nn.Conv1d and torch.nn.Conv3d
    by using convolution.

    Args:
        problem (ConvProblem): Problem for testing unfold operation.
    """
    problem.set_up()
    input = torch.rand(problem.input_shape)

    result_conv = problem.module(input)
    result_conv_by_unfold = convolution_with_unfold(input, problem.module)

    check_sizes_and_values(result_conv, result_conv_by_unfold)
