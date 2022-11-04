"""Tests BackPACK's ``SqrtGGNExact`` and ``SqrtGGNMC`` extension."""

from math import isclose
from test.automated_test import check_sizes_and_values
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import ExtensionsTestProblem, make_test_problems
from test.extensions.secondorder.sqrt_ggn.sqrt_ggn_settings import SQRT_GGN_SETTINGS
from test.utils.skip_test import skip_large_parameters, skip_subsampling_conflict
from typing import List, Union

from pytest import fixture, mark

PROBLEMS = make_test_problems(SQRT_GGN_SETTINGS)

SUBSAMPLINGS = [None, [0, 0], [2, 0]]
SUBSAMPLING_IDS = [f"subsampling={s}".replace(" ", "") for s in SUBSAMPLINGS]


@fixture(params=PROBLEMS, ids=lambda p: p.make_id())
def problem(request) -> ExtensionsTestProblem:
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


@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLING_IDS)
def test_ggn_exact(
    problem: ExtensionsTestProblem, subsampling: Union[List[int], None]
) -> None:
    """Compare exact GGN from BackPACK's matrix square root with autograd.

    Args:
        problem: Test case with small network whose GGN can be evaluated.
        subsampling: Indices of active samples. ``None`` uses the full mini-batch.
    """
    skip_large_parameters(problem)
    skip_subsampling_conflict(problem, subsampling)

    autograd_res = AutogradExtensions(problem).ggn(subsampling=subsampling)
    backpack_res = BackpackExtensions(problem).ggn(subsampling=subsampling)

    check_sizes_and_values(autograd_res, backpack_res)


@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLING_IDS)
def test_sqrt_ggn_mc_integration(
    problem: ExtensionsTestProblem, subsampling: Union[List[int], None]
) -> None:
    """Check if MC-approximated GGN matrix square root code executes.

    Note:
        This test does not perform correctness checks on the results,
        which are expensive because a large number of samples is required.
        Such a check is performed by `test_sqrt_ggn_mc`, which is run less
        frequently.

    Args:
        problem: Test case with small network whose GGN can be evaluated.
        subsampling: Indices of active samples. ``None`` uses the full mini-batch.
    """
    skip_large_parameters(problem)
    skip_subsampling_conflict(problem, subsampling)

    BackpackExtensions(problem).sqrt_ggn_mc(mc_samples=1, subsampling=subsampling)


@mark.montecarlo
@mark.parametrize("subsampling", SUBSAMPLINGS, ids=SUBSAMPLING_IDS)
def test_ggn_mc(
    problem: ExtensionsTestProblem, subsampling: Union[List[int], None]
) -> None:
    """Compare MC-approximated GGN from BackPACK with exact version from autograd.

    Args:
        problem: Test case with small network whose GGN can be evaluated.
        subsampling: Indices of active samples. ``None`` uses the full mini-batch.
    """
    skip_large_parameters(problem)
    skip_subsampling_conflict(problem, subsampling)

    autograd_res = AutogradExtensions(problem).ggn(subsampling=subsampling)
    atol, rtol = 5e-3, 5e-3
    mc_samples, chunks = 150000, 15
    backpack_res = BackpackExtensions(problem).ggn_mc(
        mc_samples, chunks=chunks, subsampling=subsampling
    )

    # compare normalized entries ∈ [-1; 1] (easier to tune atol)
    max_val = max(autograd_res.abs().max(), backpack_res.abs().max())
    # NOTE: The GGN can be exactly zero; e.g. if a ReLU after all parameters zeroes
    # its input, its Jacobian is thus zero and will cancel the backpropagated GGN
    if not isclose(max_val, 0):
        autograd_res, backpack_res = autograd_res / max_val, backpack_res / max_val

    check_sizes_and_values(autograd_res, backpack_res, atol=atol, rtol=rtol)
