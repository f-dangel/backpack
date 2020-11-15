from torch import exp, gt

from backpack.core.derivatives.elementwise import ElementwiseDerivatives


class SELUDerivatives(ElementwiseDerivatives):
    """Alpha and scale are not input_kwargs"""

    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    def hessian_is_zero(self):
        """`SELU''(x) != 0`."""
        return False

    def df(self, module, g_inp, g_out):
        """First SELU derivative: `SELU'(x) = scale if x < 0 else scale*alpha*e^x`. """

        df_SELU = gt(module.input0, 0).float()
        df_SELU[df_SELU == 1] = self.scale
        df_SELU[df_SELU == 0] = (
            self.scale * self.alpha * exp(module.input0[df_SELU == 0])
        )
        return df_SELU

    def d2f(self, module, g_inp, g_out):
        """Second SELU derivative: `SELU''(x) = 0 if x < 0 else scale*alpha*e^x`. """

        d2f_SELU = gt(module.input0, 0).float()
        d2f_SELU[d2f_SELU == 1] = 0
        d2f_SELU[d2f_SELU == 0] = (
            self.scale * self.alpha * exp(module.input0[d2f_SELU == 0])
        )
        return d2f_SELU
