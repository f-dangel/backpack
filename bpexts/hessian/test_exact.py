"""Test exact computation of Hessian."""

from torch import tensor
from .exact import exact_hessian
from ..utils import torch_allclose


def test_exact_hessian_simple():
    """Test exact Hessian computation for a simple function."""
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
    # expected outcome
    result = tensor([[1, 4, 5],
                     [4, 2, 6],
                     [5, 6, 3]]).float()
    # compute and compare results
    hessian = exact_hessian(f, parameters, show_progress=True)
    assert torch_allclose(hessian, result)
