from torch import gt
from torch.nn import ReLU
from .elementwise import ElementwiseDerivatives


class ReLUDerivatives(ElementwiseDerivatives):
    def get_module(self):
        return ReLU

    def jac_t_mat_prod(self, module, grad_input, grad_output, mat):
        d_relu = gt(module.input0, 0).float()
        return super().jac_t_mat_prod(d_relu, mat)

    def jac_mat_prod(self, module, grad_input, grad_output, mat):
        d_relu = gt(module.input0, 0).float()
        return super().jac_mat_prod(d_relu, mat)

    def hessian_is_zero(self):
        return True
