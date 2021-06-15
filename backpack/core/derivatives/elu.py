from torch import exp, gt

from backpack.core.derivatives.elementwise import ElementwiseDerivatives


class ELUDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self):
        """`ELU''(x) ≠ 0`."""
        return False

    def df(self, module, g_inp, g_out):
        """First ELU derivative: `ELU'(x) = alpha * e^x if x <= 0 else 1`."""
        pos = gt(module.input0, 0)
        dtype = module.input0.dtype

        return pos.to(dtype) + module.alpha * pos.logical_not().to(dtype) * exp(
            module.input0
        )

    def d2f(self, module, g_inp, g_out):
        """Second ELU derivative: `ELU''(x) = alpha * e^x if x <= 0 else 0`."""
        return self.df(module, g_inp, g_out) - gt(module.input0, 0).to(
            module.input0.dtype
        )
