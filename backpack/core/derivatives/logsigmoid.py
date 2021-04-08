from torch import exp

from backpack.core.derivatives.elementwise import ElementwiseDerivatives


class LogSigmoidDerivatives(ElementwiseDerivatives):
    def hessian_is_zero(self):
        """`logsigmoid''(x) ≠ 0`."""
        return False

    def df(self, module, g_inp, g_out):
        """First Logsigmoid derivative: `logsigmoid'(x) = 1 / (e^x + 1) `."""
        return 1 / (exp(module.input0) + 1)

    def d2f(self, module, g_inp, g_out):
        """Second Logsigmoid derivative: `logsigmoid''(x) = - e^x / (e^x + 1)^2`."""
        exp_input = exp(module.input0)
        return -(exp_input / (exp_input + 1) ** 2)
