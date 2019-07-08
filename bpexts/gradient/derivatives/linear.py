from ...utils import einsum
from torch.nn import Linear
from .basederivatives import BaseDerivatives


class LinearDerivatives(BaseDerivatives):
    def get_module(self):
        return Linear

    def jac_t_mat_prod(self, module, grad_input, grad_output, mat):
        d_linear = module.weight
        return einsum('ij,bic->bjc', (d_linear, mat))

    def hessian_is_zero(self):
        return True
