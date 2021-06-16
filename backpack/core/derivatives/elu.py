"""Partial derivatives for the ELU activation function."""
from torch import exp, le, ones_like, zeros_like

from backpack.core.derivatives.elementwise import ElementwiseDerivatives


class ELUDerivatives(ElementwiseDerivatives):
    """Implement first- and second-order partial derivatives of ELU."""

    def hessian_is_zero(self):
        """`ELU''(x) ≠ 0`."""
        return False

    def df(self, module, g_inp, g_out):
        """First ELU derivative: `ELU'(x) = alpha * e^x if x <= 0 else 1`."""
        non_pos = le(module.input0, 0)

        result = ones_like(module.input0)
        result[non_pos] = module.alpha * exp(module.input0[non_pos])

        return result

    def d2f(self, module, g_inp, g_out):
        """Second ELU derivative: `ELU''(x) = alpha * e^x if x <= 0 else 0`."""
        non_pos = le(module.input0, 0)

        result = zeros_like(module.input0)
        result[non_pos] = module.alpha * exp(module.input0[non_pos])

        return result
