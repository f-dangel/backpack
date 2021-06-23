"""Contains batch_l2 extension for Linear."""
from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from torch import Tensor, einsum
from torch.nn import Linear

from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.firstorder.batch_l2_grad.batch_l2_base import BatchL2Base

if TYPE_CHECKING:
    from backpack.extensions import BatchL2Grad


class BatchL2Linear(BatchL2Base):
    """batch_l2 extension for Linear."""

    def __init__(self):
        """Initialization."""
        super().__init__(["bias", "weight"], derivatives=LinearDerivatives())

    def weight(
        self,
        ext: BatchL2Grad,
        module: Linear,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        backproped: None,
    ) -> Tensor:
        """batch_l2 for weight.

        Args:
            ext: extension
            module: module
            g_inp: input gradients
            g_out: output gradients
            backproped: backpropagation quantities

        Returns:
            batch_l2 for weight
        """
        return einsum("ni,nj->n", g_out[0] ** 2, module.input0 ** 2)
