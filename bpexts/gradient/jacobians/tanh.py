from torch.nn import Tanh
from .elementwise import ElementwiseJacobian


class TanhJacobian(ElementwiseJacobian):

    def get_module(self):
        return Tanh

    def jac_mat_prod(self, module, grad_input, grad_output, mat):
        d_tanh = 1. - module.output**2
        return super().jac_mat_prod(d_tanh, mat)
