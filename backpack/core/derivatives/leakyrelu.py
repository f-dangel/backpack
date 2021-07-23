"""Partial derivatives for the leaky ReLU layer."""
from typing import List, Tuple

from torch import Tensor, gt
from torch.nn import LeakyReLU

from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from backpack.utils.subsampling import subsample


class LeakyReLUDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self, module: LeakyReLU) -> bool:
        """`LeakyReLU''(x) = 0`."""
        return True

    def df(
        self,
        module: LeakyReLU,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ) -> Tensor:
        """``LeakyReLU'(x) = negative_slope if x < 0 else 1``."""
        input0 = subsample(module.input0, subsampling=subsampling)
        df_leakyrelu = gt(input0, 0).to(input0.dtype)
        df_leakyrelu[df_leakyrelu == 0] = module.negative_slope
        return df_leakyrelu
