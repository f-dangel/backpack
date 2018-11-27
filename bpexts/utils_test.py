"""Test utility functions."""

from torch import Tensor
from .utils import torch_allclose


def test_torch_allclose():
    """Test equality between two tensors within a tolerance."""
    t1 = Tensor([1, 2, 3]).float()
    t2 = Tensor([0.9999, 2, 3])
    assert torch_allclose(t1, t2, rtol=0, atol=1E-3)
    assert not torch_allclose(t1, t2, rtol=0, atol=1E-5)
