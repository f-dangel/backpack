from torch import gt
from torch.nn import LeakyReLU

from backpack.core.derivatives.elementwise import ElementwiseDerivatives


class LeakyReLUDerivatives(ElementwiseDerivatives):
    def get_module(self):
        return LeakyReLU

    def hessian_is_zero(self):
        """`LeakyReLU''(x) = 0`."""
        return True

    def df(self, module, g_inp, g_out):
        """First LeakyReLU derivative:
        `LeakyReLU'(x) = negative_slope if x < 0 else 1`."""
        df_leakyrelu = gt(module.input0, 0).float()
        df_leakyrelu[df_leakyrelu == 0] = module.negative_slope
        return df_leakyrelu
