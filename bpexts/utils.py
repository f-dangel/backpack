"""Utility functions."""

from numpy import allclose


def torch_allclose(a, b, *args, **kwargs):
    """True if two tensors are element-wise equal within a tolerance.

    Internally, `numpy.allclose` is called.
    """
    return allclose(a.data, b.data, *args, **kwargs)
