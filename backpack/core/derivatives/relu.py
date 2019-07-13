from torch import gt
from torch.nn import ReLU
from .elementwise import ElementwiseDerivatives


class ReLUDerivatives(ElementwiseDerivatives):
    def get_module(self):
        return ReLU

    def hessian_is_zero(self):
        return True

    def df(self, module, grad_input, grad_output):
        return gt(module.input0, 0).float()
