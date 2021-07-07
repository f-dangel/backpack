"""Test BackPACK's ``BatchGrad`` extension."""
from test.automated_test import check_sizes_and_values
from test.extensions.firstorder.batch_grad.batch_grad_settings import (
    BATCH_GRAD_SETTINGS,
)
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import ExtensionsTestProblem, make_test_problems
from test.extensions.utils import skip_if_subsampling_conflict
from typing import List, Union

from pytest import fixture, mark

PROBLEMS = make_test_problems(BATCH_GRAD_SETTINGS)

SUBSAMPLINGS = [None, [0, 0], [2, 0]]
SUBSAMPLING_IDS = [f"subsampling={s}".replace(" ", "") for s in SUBSAMPLINGS]


@fixture(params=PROBLEMS, ids=lambda p: p.make_id())
def problem(request) -> ExtensionsTestProblem:
    """Set up and tear down a test case.

    Args:
        request: Pytest request.

    Yields:
        Instantiated test case.
    """
    problem = request.param
    problem.set_up()
    yield problem
    problem.tear_down()


@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLING_IDS)
def test_batch_grad(
    problem: ExtensionsTestProblem, subsampling: Union[List[int], None]
) -> None:
    """Test individual gradients.

    Args:
        problem: Test case.
        subsampling: Indices of active samples.
    """
    skip_if_subsampling_conflict(problem, subsampling)

    backpack_res = BackpackExtensions(problem).batch_grad(subsampling)
    autograd_res = AutogradExtensions(problem).batch_grad(subsampling)

    check_sizes_and_values(autograd_res, backpack_res)
