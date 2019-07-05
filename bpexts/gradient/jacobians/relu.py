from torch import gt
from torch.nn import ReLU
from .elementwise import ElementwiseJacobian


class ReLUJacobian(ElementwiseJacobian):

    def get_module(self):
        return ReLU

    def jac_mat_prod(self, module, grad_input, grad_output, mat):
        d_relu = gt(module.input0, 0).float()
        return super().jac_mat_prod(d_relu, mat)
