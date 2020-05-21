from torch import gt, exp
from torch.nn import SELU

from backpack.core.derivatives.elementwise import ElementwiseDerivatives


class SELUDerivatives(ElementwiseDerivatives):
    def get_module(self):
        return SELU

    def hessian_is_zero(self):
        """`SELU''(x) != 0`."""
        return False

    def df(self, module, g_inp, g_out):
        """First SELU derivative: `SELU'(x) = scale if x < 0 else scale*alpha*e^x`. """
        """Alpha and scale are not input_kwargs"""
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946

        df_SELU = gt(module.input0, 0).float()
        df_SELU[df_SELU == 1] = scale
        df_SELU[df_SELU == 0] = scale * alpha * exp(module.input0[df_SELU == 0])
        return df_SELU

    def d2f(self, module, g_inp, g_out):
        """First SELU derivative: `SELU'(x) = 0 if x < 0 else scale*alpba(e^x`. """
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946

        df_SELU = gt(module.input0, 0).float()
        df_SELU[df_SELU == 1] = 0
        df_SELU[df_SELU == 0] = scale * alpha * exp(module.input0[df_SELU == 0])
        return df_SELU
