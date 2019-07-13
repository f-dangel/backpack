from torch.nn import Tanh
from .elementwise import ElementwiseDerivatives


class TanhDerivatives(ElementwiseDerivatives):
    def get_module(self):
        return Tanh

    def hessian_is_zero(self):
        return False

    def hessian_is_diagonal(self):
        return True

    def df(self, module, grad_input, grad_output):
        return 1. - module.output**2

    def d2f(self, module, grad_input, grad_output):
        return (-2. * module.output * (1. - module.output**2))
