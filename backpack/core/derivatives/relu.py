from torch import gt
from torch.nn import ReLU
from .elementwise import ElementwiseDerivatives


class ReLUDerivatives(ElementwiseDerivatives):
    def get_module(self):
        return ReLU

    def hessian_is_zero(self):
        return True

    def df(self, module, g_inp, g_out):
        return gt(module.input0, 0).float()
