from torch import eq
from torch.nn import Dropout
from .elementwise import ElementwiseDerivatives


class DropoutDerivatives(ElementwiseDerivatives):
    def get_module(self):
        return Dropout

    def hessian_is_zero(self):
        return True

    def df(self, module, grad_input, grad_output):
        scaling = 1 / (1 - module.p)
        mask = 1 - eq(grad_input, 0.).float()
        return mask * scaling
