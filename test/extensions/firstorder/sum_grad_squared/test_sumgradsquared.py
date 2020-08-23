"""Test class for module Batch_grad (batch gradients) 
from `backpack.core.extensions.firstorder`

Test individual gradients for the following layers:
- batch gradients of linear layers
- batch gradients of convolutional layers

"""
from test.automated_test import check_sizes_and_values
from test.extensions.problem import make_test_problems
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.firstorder.sum_grad_squared.sumgradsquared_settings import (
    SUMGRADSQUARED_SETTINGS,
)

import pytest


PROBLEMS = make_test_problems(SUMGRADSQUARED_SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_sum_grad_squared(problem):
    """Test sum of individual gradients and square them

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
    """
    problem.set_up()

    backpack_res = BackpackExtensions(problem).sgs()
    autograd_res = AutogradExtensions(problem).sgs()

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()
