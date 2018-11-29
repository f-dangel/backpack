"""Test exact computation of Hessian."""

from torch import tensor
from .exact import exact_hessian
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


def involved_function_with_parameters():
    """Return complicated example function with parameters."""
    # parameters
    x = tensor([[1, -2],
                [3, -4]], requires_grad=True).float()
    y = tensor([5, 6], requires_grad=True).float()
    parameters = [x, y]
    # input (no parameter)
    input = tensor([1, 2]).float()
    # composition of functions
    temp = x.matmul(input) + y
    temp_result = tensor([2, 1]).float()
    assert torch_allclose(temp, temp_result)
    f = (temp**2).sum()
    f_result = tensor([5])
    assert torch_allclose(f, f_result)
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


def test_exact_hessian_involved():
    """Test exact Hessian computation for a complicated function."""
    f, parameters = involved_function_with_parameters()
    # expected outcome
    result = tensor([[2, 4, 0, 0, 2, 0],
                     [4, 8, 0, 0, 4, 0],
                     [0, 0, 2, 4, 0, 2],
                     [0, 0, 4, 8, 0, 4],
                     [2, 4, 0, 0, 2, 0],
                     [0, 0, 2, 4, 0, 2]]).float()
    # compute and compare results
    print(result)
    hessian = exact_hessian(f, parameters, show_progress=True)
    print(hessian)
    assert torch_allclose(hessian, result)
