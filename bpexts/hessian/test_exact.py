"""Test exact computation of Hessian."""

from torch import tensor
from .exact import (exact_hessian, exact_hessian_diagonal_blocks)
from ..utils import torch_allclose


def simple_function_with_parameters():
    """Return simple example function with parameters."""
    # parameters
    x = tensor([1, 1], requires_grad=True).float()
    y = tensor([1], requires_grad=True).float()
    parameters = [x, y]
    # function (Hessian easily computable)
    f = 0.5 * (x[0]**2
               + 2 * x[1]**2
               + 3 * y[0]**2)\
        + 4 * x[0] * x[1]\
        + 5 * x[0] * y[0]\
        + 6 * x[1] * y[0]
    return f, parameters


def test_exact_hessian_simple():
    """Test exact Hessian computation for a simple function."""
    f, parameters = simple_function_with_parameters()
    # expected outcome
    result = tensor([[1, 4, 5],
                     [4, 2, 6],
                     [5, 6, 3]]).float()
    # compute and compare results
    hessian = exact_hessian(f, parameters, show_progress=True)
    assert torch_allclose(hessian, result)


def test_exact_hessian_diagonal_blocks_simple():
    """Test exact block Hessian computation for a simple function."""
    f, parameters = simple_function_with_parameters()
    # expected outcome
    block1 = tensor([[1, 4],
                     [4, 2]]).float()
    block2 = tensor([3]).float()
    results = [block1, block2]
    # compute and compare
    hessian_blocks = exact_hessian_diagonal_blocks(f, parameters,
                                                   show_progress=True)
    for block, result in zip(hessian_blocks, results):
        assert torch_allclose(block, result)
