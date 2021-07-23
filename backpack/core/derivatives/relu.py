"""Partial derivatives for the ReLU activation function."""
from typing import List, Tuple

from torch import Tensor, gt
from torch.nn import ReLU

from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from backpack.utils.subsampling import subsample


class ReLUDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self, module):
        """`ReLU''(x) = 0`."""
        return True

    def df(
        self,
        module: ReLU,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ) -> Tensor:
        """First ReLU derivative: `ReLU'(x) = 0 if x < 0 else 1`."""
        input0 = subsample(module.input0, subsampling=subsampling)
        return gt(input0, 0).to(input0.dtype)
