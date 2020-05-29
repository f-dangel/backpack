from torch import gt, exp
from torch.nn import ELU

from backpack.core.derivatives.elementwise import ElementwiseDerivatives


class ELUDerivatives(ElementwiseDerivatives):
    def get_module(self):
        return ELU

    def hessian_is_zero(self):
        """`ELU''(x) ≠ 0`."""
        return False

    def df(self, module, g_inp, g_out):
        """First ELU derivative: `ELU'(x) = alpha * e^x if x < 0 else 1`. """
        df_ELU = gt(module.input0, 0).float()
        df_ELU[df_ELU == 0] = module.alpha * exp(module.input0[df_ELU == 0])
        return df_ELU

    def d2f(self, module, g_inp, g_out):
        """Second ELU derivative: `ELU''(x) = alpha * e^x if x < 0 else 1`. """
        return self.df(module, g_inp, g_out)
