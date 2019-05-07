"""Test batch-averaged Hessian computation."""

from torch import tensor
from .loss import batch_summed_hessian
from ..utils import torch_allclose


def example_loss_function(x):
    """Sum up all square elements of a tensor."""
    return (x**2).view(-1).sum(0)


def test_batch_averaged_hessian():
    """Test batch-averaged Hessian."""
    # 2 samples of 3d vectors
    x = tensor([[1., 2., 3.], [4., 5., 6.]], requires_grad=True)
    f = example_loss_function(x)
    result = tensor([[4., 0., 0.], [0., 4., 0.], [0., 0., 4.]])
    avg_hessian = batch_summed_hessian(f, x)
    assert torch_allclose(result, avg_hessian)
