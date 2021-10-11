"""Tests whether batch axis is always first."""
from pytest import raises

from backpack.custom_module.permute import Permute


def test_permute_batch_axis() -> None:
    """Verify that an Error is raised in the correct settings."""
    Permute(0, 1, 2)
    Permute(0, 2, 1)
    Permute(0, 2, 3, 1)
    with raises(ValueError):
        Permute(1, 0, 2)
    with raises(ValueError):
        Permute(2, 0, 1)

    Permute(1, 2, init_transpose=True)
    Permute(3, 1, init_transpose=True)
    Permute(2, 1, init_transpose=True)
    with raises(ValueError):
        Permute(0, 1, init_transpose=True)
    with raises(ValueError):
        Permute(1, 0, init_transpose=True)
    with raises(ValueError):
        Permute(2, 0, init_transpose=True)
