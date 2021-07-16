"""Partial derivatives for the dropout layer."""
from typing import List, Tuple

from torch import Tensor, eq
from torch.nn import Dropout

from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from backpack.utils.subsampling import subsample


class DropoutDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self, module: Dropout) -> bool:
        """``Dropout''(x) = 0``."""
        return True

    def df(
        self,
        module: Dropout,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ) -> Tensor:
        output = subsample(module.output, subsampling=subsampling)
        scaling = 1 / (1 - module.p)
        mask = 1 - eq(output, 0.0).float()
        return mask * scaling
