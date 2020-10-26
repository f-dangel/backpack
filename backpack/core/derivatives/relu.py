from torch import gt
from torch.nn import ReLU

from backpack.core.derivatives.elementwise import ElementwiseDerivatives


class ReLUDerivatives(ElementwiseDerivatives):
    def get_module(self):
        return ReLU

    def hessian_is_zero(self):
        """`ReLU''(x) = 0`."""
        return True

    def df(self, module, g_inp, g_out):
        """First ReLU derivative: `ReLU'(x) = 0 if x < 0 else 1`. """
        return gt(module.input0, 0).float()
