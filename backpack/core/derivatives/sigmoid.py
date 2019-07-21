from torch.nn import Sigmoid
from .elementwise import ElementwiseDerivatives


class SigmoidDerivatives(ElementwiseDerivatives):
    def get_module(self):
        return Sigmoid

    def hessian_is_zero(self):
        return False

    def hessian_is_diagonal(self):
        return True

    def df(self, module, g_inp, g_out):
        return module.output * (1. - module.output)

    def d2f(self, module, g_inp, g_out):
        return module.output * (1 - module.output) * (1 - 2 * module.output)
