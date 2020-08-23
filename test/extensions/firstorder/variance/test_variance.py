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
from test.extensions.firstorder.variance.variance_settings import VARIANCE_SETTINGS

import pytest


PROBLEMS = make_test_problems(VARIANCE_SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_variance(problem):
    """Test l2 norm of individual gradients

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
    """
    problem.set_up()

    backpack_res = BackpackExtensions(problem).variance()
    autograd_res = AutogradExtensions(problem).variance()

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()
