from torch.nn import Tanh
from .elementwise import ElementwiseDerivatives


class TanhDerivatives(ElementwiseDerivatives):

    def get_module(self):
        return Tanh

    def jac_mat_prod(self, module, grad_input, grad_output, mat):
        d_tanh = 1. - module.output**2
        return super().jac_mat_prod(d_tanh, mat)

    def hessian_is_zero(self):
        return False

    def hessian_is_diagonal(self):
        return True

    def hessian_diagonal(self, module, grad_input, grad_output):
        out = module.output
        d2_tanh = (-2. * out * (1. - out**2))
        return super().hessian_diagonal(d2_tanh, grad_output[0])
