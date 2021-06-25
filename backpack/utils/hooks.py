"""Utility functions to handle the backpropagation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from torch import Tensor
from torch.nn import Module

if TYPE_CHECKING:
    from backpack import BackpropExtension


def no_op(*args, **kwargs):
    """Placeholder function that accepts arbitrary input and does nothing."""
    return None


def apply_no_op(
    ext: BackpropExtension,
    module: Module,
    g_inp: Tuple[Tensor],
    g_out: Tuple[Tensor],
) -> None:
    pass
