"""Partial derivatives for the Tanh activation function."""
from typing import List, Tuple

from torch import Tensor
from torch.nn import Tanh

from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from backpack.utils.subsampling import subsample


class TanhDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self, module):
        return False

    def df(
        self,
        module: Tanh,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ) -> Tensor:
        output = subsample(module.output, subsampling=subsampling)
        return 1.0 - output**2

    def d2f(self, module, g_inp, g_out):
        return -2.0 * module.output * (1.0 - module.output**2)
