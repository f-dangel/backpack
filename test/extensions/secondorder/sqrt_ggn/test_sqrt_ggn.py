"""Tests BackPACK's ``SqrtGGNExact`` extension."""

from test.automated_test import check_sizes_and_values
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import ExtensionsTestProblem, make_test_problems
from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS

from pytest import fixture, skip

PROBLEMS = make_test_problems(SECONDORDER_SETTINGS)
IDS = [problem.make_id() for problem in PROBLEMS]


@fixture(params=PROBLEMS, ids=IDS)
def problem(request, max_num_params: int = 4000) -> ExtensionsTestProblem:
    """Set seed, create tested model, loss, data. Finally clean up.

    Models with too many parameters are skipped.

    Args:
        request (SubRequest): Request for the fixture from a test/fixture function.
        max_num_params: Maximum number of model parameters to run the case.
            Default: ``4000``.

    Yields:
        Test case with deterministically constructed attributes.
    """
    case = request.param
    case.set_up()

    num_params = sum(p.numel() for p in case.model.parameters() if p.requires_grad)
    if num_params <= max_num_params:
        yield case
    else:
        skip(f"Model has too many parameters: {num_params} > {max_num_params}")

    case.tear_down()


def test_sqrt_ggn_exact(problem: ExtensionsTestProblem):
    """Compare exact GGN from BackPACK's matrix square root with autograd.

    Args:
        problem: Test case
    """
    autograd_res = AutogradExtensions(problem).ggn()
    backpack_res = BackpackExtensions(problem).ggn()

    check_sizes_and_values(autograd_res, backpack_res)
