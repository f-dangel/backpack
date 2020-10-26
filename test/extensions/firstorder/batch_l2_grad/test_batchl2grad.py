"""Test class for module Batch_l2_grad (L2 norm of batch gradients) 
from `backpack.core.extensions.firstorder`

Test individual gradients for the following layers:
- L2 norm of batch gradients of linear layers
- L2 norm of batch gradients of convolutional layers

"""
from test.automated_test import check_sizes_and_values
from test.extensions.problem import make_test_problems
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.firstorder.batch_l2_grad.batchl2grad_settings import (
    BATCHl2GRAD_SETTINGS,
)

import pytest


PROBLEMS = make_test_problems(BATCHl2GRAD_SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_batch_l2_grad(problem):
    """Test l2 norm of individual gradients

    Args:
        problem (ExtensionsTestProblem): Problem for extension test.
    """
    problem.set_up()

    backpack_res = BackpackExtensions(problem).batch_l2_grad()
    autograd_res = AutogradExtensions(problem).batch_l2_grad()

    check_sizes_and_values(autograd_res, backpack_res)
    problem.tear_down()
