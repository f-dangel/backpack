"""Partial derivatives for the SELU activation function."""
from typing import List, Tuple

from torch import Tensor, exp, le, ones_like, zeros_like
from torch.nn import SELU

from backpack.core.derivatives.elementwise import ElementwiseDerivatives
from backpack.utils.subsampling import subsample


class SELUDerivatives(ElementwiseDerivatives):
    """Implement first- and second-order partial derivatives of SELU."""

    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    def hessian_is_zero(self, module):
        """`SELU''(x) != 0`."""
        return False

    def df(
        self,
        module: SELU,
        g_inp: Tuple[Tensor],
        g_out: Tuple[Tensor],
        subsampling: List[int] = None,
    ) -> Tensor:
        """First SELU derivative: `SELU'(x) = scale if x > 0 else scale*alpha*e^x`."""
        input0 = subsample(module.input0, subsampling=subsampling)
        non_pos = le(input0, 0)

        result = self.scale * ones_like(input0)
        result[non_pos] = self.scale * self.alpha * exp(input0[non_pos])

        return result

    def d2f(self, module, g_inp, g_out):
        """Second SELU derivative: `SELU''(x) = 0 if x > 0 else scale*alpha*e^x`."""
        non_pos = le(module.input0, 0)

        result = zeros_like(module.input0)
        result[non_pos] = self.scale * self.alpha * exp(module.input0[non_pos])

        return result
