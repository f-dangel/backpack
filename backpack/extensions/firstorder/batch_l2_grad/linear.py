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
        has_additional_axes = g_out[0].dim() > 2

        if has_additional_axes:
            # TODO Compare `torch.einsum`, `opt_einsum.contract` and the base class
            # implementation: https://github.com/fKunstner/backpack-discuss/issues/111
            dE_dY = g_out[0].flatten(start_dim=1, end_dim=-2)
            X = module.input0.flatten(start_dim=1, end_dim=-2)
            return einsum("nmi,nmj,nki,nkj->n", dE_dY, X, dE_dY, X)
        else:
            return einsum("ni,nj->n", g_out[0] ** 2, module.input0**2)
