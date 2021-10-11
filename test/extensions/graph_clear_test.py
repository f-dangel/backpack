"""Test whether the graph is clear after a backward pass."""
from typing import Tuple

from pytest import fixture
from torch import Tensor, rand, rand_like
from torch.nn import Flatten, Linear, Module, MSELoss, ReLU, Sequential

from backpack import backpack, extend
from backpack.extensions import DiagGGNExact

PROBLEM_STRING = ["standard", "flatten_no_op", "flatten_with_op"]


def test_graph_clear(problem) -> None:
    """Test that the graph is clear after a backward pass.

    More specifically, test that there are no saved quantities left over.

    Args:
        problem: problem consisting of inputs, and model
    """
    inputs, model = problem
    extension = DiagGGNExact()
    outputs = extend(model)(inputs)
    loss = extend(MSELoss())(outputs, rand_like(outputs))
    with backpack(extension):
        loss.backward()

    # test that the dictionary is empty
    saved_quantities: dict = extension.saved_quantities._saved_quantities
    assert type(saved_quantities) is dict
    assert not saved_quantities


@fixture(params=PROBLEM_STRING, ids=PROBLEM_STRING)
def problem(request) -> Tuple[Tensor, Module]:
    """Problem setting.

    Args:
        request: pytest request, contains parameters

    Yields:
        inputs and model

    Raises:
        NotImplementedError: if problem string is unknown
    """
    batch_size, in_dim, out_dim = 2, 3, 4
    inputs = rand(batch_size, in_dim)
    if request.param == PROBLEM_STRING[0]:
        model = Sequential(Linear(in_dim, out_dim), ReLU(), Linear(out_dim, out_dim))
    elif request.param == PROBLEM_STRING[1]:
        model = Sequential(Linear(in_dim, out_dim), Flatten(), Linear(out_dim, out_dim))
    elif request.param == PROBLEM_STRING[2]:
        inputs = rand(batch_size, in_dim, in_dim)
        model = Sequential(
            Linear(in_dim, out_dim), Flatten(), Linear(in_dim * out_dim, out_dim)
        )
    else:
        raise NotImplementedError(f"unknown request.param={request.param}")
    yield inputs, model
