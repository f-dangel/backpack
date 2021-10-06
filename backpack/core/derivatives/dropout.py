"""Partial derivatives for the dropout layer."""
from typing import List, Tuple

from torch import Tensor, eq, ones_like
from torch.nn import Dropout

from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from backpack.utils.subsampling import subsample


class DropoutDerivatives(ElementwiseDerivatives):
    """Derivatives for the Dropout module."""

    def hessian_is_zero(self, module: Dropout) -> bool:
        """``Dropout''(x) = 0``.

        Args:
            module: dropout module

        Returns:
            whether hessian is zero
        """
        return True

    def df(
        self,
        module: Dropout,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ) -> Tensor:  # noqa: D102
        output = subsample(module.output, subsampling=subsampling)
        if module.training:
            scaling = 1 / (1 - module.p)
            mask = 1 - eq(output, 0.0).to(output.dtype)
            return mask * scaling
        else:
            return ones_like(output)
