"""Tests BackPACK's ``GGNMP`` extension."""

from test.automated_test import check_sizes_and_values
from test.extensions.curvmatprod.ggnmp.ggnmp_settings import GGNMP_SETTINGS
from test.extensions.implementation.autograd import AutogradExtensions
from test.extensions.implementation.backpack import BackpackExtensions
from test.extensions.problem import ExtensionsTestProblem, make_test_problems
from typing import Iterable, List

from pytest import fixture
from torch import Tensor, rand_like, stack

PROBLEMS = make_test_problems(GGNMP_SETTINGS)


@fixture(params=PROBLEMS, ids=lambda p: p.make_id())
def instantiated_problem(request) -> ExtensionsTestProblem:
    """Set seed, create tested network, loss, and data. Finally clean up.

    Args:
        request (SubRequest): Request for the fixture from a test/fixture function.

    Yields:
        Test case with deterministically constructed attributes.
    """
    case = request.param
    case.set_up()
    yield case
    case.tear_down()


def create_mat_list(parameters: Iterable[Tensor], V: int) -> List[Tensor]:
    """Create the parameter-wise matrices that serve as input to the curvature product.

    Args:
        parameters: An iterator over the parameters of a neural net.
        V: Number of matrix columns

    Returns:
        List of parameter-wise matrices that serve as input to multiplication with
        the respective curvature matrix block.
    """
    return [stack([rand_like(p) for _ in range(V)]) for p in parameters]


def test_ggnmp(instantiated_problem: ExtensionsTestProblem, V=3) -> None:
    """Test the block diagonal GGN-matrix product.

    Args:
        instantiated_problem: Instantiated test case.
        V: Number of vectorized GGN multiplications. Default: ``3``.
    """
    mat_list = create_mat_list(instantiated_problem.model.parameters(), V)

    backpack_res = BackpackExtensions(instantiated_problem).ggnmp(mat_list)
    autograd_res = AutogradExtensions(instantiated_problem).ggnmp(mat_list)

    check_sizes_and_values(autograd_res, backpack_res)


def create_vec_list(parameters: Iterable[Tensor]) -> List[Tensor]:
    """Create parameter-wise vectors that serve as input to the curvature product.

    Args:
        parameters: An iterator over the parameters of a neural net.

    Returns:
        List of parameter-wise vectors that serve as input to multiplication with
        the respective curvature matrix block.
    """
    return [rand_like(p) for p in parameters]


def test_ggnmp_accept_vectors(instantiated_problem: ExtensionsTestProblem) -> None:
    """Test whether the block diagonal GGN-matrix product works with vector inputs.

    Args:
        instantiated_problem: Instantiated test case.
    """
    vec_list = create_vec_list(instantiated_problem.model.parameters())

    backpack_res = BackpackExtensions(instantiated_problem).ggnmp(vec_list)
    autograd_res = AutogradExtensions(instantiated_problem).ggnvp(vec_list)

    check_sizes_and_values(autograd_res, backpack_res)
