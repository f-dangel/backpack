"""Utility functions."""

from numpy import allclose
from numpy.random import seed as numpy_seed
from torch import manual_seed as torch_manual_seed
from torch.cuda import manual_seed as torch_cuda_manual_seed
from random import seed as random_seed


def torch_allclose(a, b, *args, **kwargs):
    """True if two tensors are element-wise equal within a tolerance.

    Internally, `numpy.allclose` is called.
    """
    return allclose(a.data, b.data, *args, **kwargs)


def set_seeds(seed):
    """Set random seeds of pyTorch (+CUDA), NumPy and random modules."""
    # PyTorch
    torch_manual_seed(seed)
    torch_cuda_manual_seed(seed)
    # NumPy
    numpy_seed(seed)
    # random
    random_seed(seed)
