"""Tests BackPACK's ``SqrtGGNExact`` and ``SqrtGGNMC`` extension."""

from test.automated_test import check_sizes_and_values
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import ExtensionsTestProblem, make_test_problems
from test.extensions.secondorder.secondorder_settings import SECONDORDER_SETTINGS

from pytest import fixture, mark, skip

PROBLEMS = make_test_problems(SECONDORDER_SETTINGS)


@fixture(params=PROBLEMS, ids=lambda p: p.make_id())
def instantiated_problem(request) -> ExtensionsTestProblem:
    """Set seed, create tested model, loss, data. Finally clean up.

    Args:
        request (SubRequest): Request for the fixture from a test/fixture function.

    Yields:
        Test case with deterministically constructed attributes.
    """
    case = request.param
    case.set_up()
    yield case
    case.tear_down()


@fixture
def small_problem(
    instantiated_problem: ExtensionsTestProblem, max_num_params=4000
) -> ExtensionsTestProblem:
    """Skip architectures with too many parameters whose GGN is expensive to evaluate.

    Args:
        instantiated_problem: Test case with instantiated model, data, etc.
        max_num_params: Maximum number of model parameters to run the case.
            Default: ``4000``.

    Yields:
        Instantiated test case whose model's are small enough.
    """
    num_params = sum(
        p.numel() for p in instantiated_problem.model.parameters() if p.requires_grad
    )
    if num_params <= max_num_params:
        yield instantiated_problem
    else:
        skip(f"Model has too many parameters: {num_params} > {max_num_params}")


def test_sqrt_ggn_exact(small_problem: ExtensionsTestProblem):
    """Compare exact GGN from BackPACK's matrix square root with autograd.

    Args:
        small_problem: Test case with small network whose GGN can be evaluated.
    """
    autograd_res = AutogradExtensions(small_problem).ggn()
    backpack_res = BackpackExtensions(small_problem).ggn()

    check_sizes_and_values(autograd_res, backpack_res)


@mark.montecarlo
def test_sqrt_ggn_mc(small_problem: ExtensionsTestProblem):
    """Compare MC-approximated GGN from BackpACK's with exact version from autograd.

    Args:
        small_problem: Test case with small network whose GGN can be evaluated.
    """
    autograd_res = AutogradExtensions(small_problem).ggn()
    atol, rtol = 5e-4, 1e-2
    mc_samples, chunks = 500000, 50
    backpack_res = BackpackExtensions(small_problem).ggn_mc(mc_samples, chunks=chunks)

    check_sizes_and_values(autograd_res, backpack_res, atol=atol, rtol=rtol)
