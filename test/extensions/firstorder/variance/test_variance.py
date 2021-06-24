"""Test BackPACK's ``Variance`` extension."""
from test.automated_test import check_sizes_and_values
from test.extensions.firstorder.variance.variance_settings import VARIANCE_SETTINGS
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import ExtensionsTestProblem, make_test_problems

import pytest

PROBLEMS = make_test_problems(VARIANCE_SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@pytest.mark.parametrize("problem", PROBLEMS, ids=IDS)
def test_variance(problem: ExtensionsTestProblem) -> None:
    """Test variance of individual gradients.

    Args:
        problem: Test case.
    """
    problem.set_up()

    backpack_res = BackpackExtensions(problem).variance()
    autograd_res = AutogradExtensions(problem).variance()

    rtol = 5e-5
    check_sizes_and_values(autograd_res, backpack_res, rtol=rtol)
    problem.tear_down()
