from torch import eq

from backpack.core.derivatives.elementwise import ElementwiseDerivatives


class DropoutDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self):
        return True

    def df(self, module, g_inp, g_out):
        scaling = 1 / (1 - module.p)
        mask = 1 - eq(module.output, 0.0).float()
        return mask * scaling
