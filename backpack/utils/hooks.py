"""Utility functions to handle the backpropagation."""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from torch import Tensor
from torch.nn import Module

if TYPE_CHECKING:
    from backpack import BackpropExtension


def no_op(*args, **kwargs):
    """Placeholder function that accepts arbitrary input and does nothing.

    Args:
        *args: anything
        **kwargs: anything
    """
    pass


def no_op_apply(
    extension: BackpropExtension,
    module: Module,
    grad_inp: Tuple[Tensor],
    grad_out: Tuple[Tensor],
) -> None:
    """This is the equivalent function of ModuleExtension::apply.

    It is used when no hooks are installed (e.g. containers).

    Args:
        extension: backpack extension
        module: module without hooks
        grad_inp: input gradients
        grad_out: output gradients
    """
    pass
