from torch.nn import Sigmoid
from .elementwise import ElementwiseDerivatives


class SigmoidDerivatives(ElementwiseDerivatives):
    def get_module(self):
        return Sigmoid

    def jac_t_mat_prod(self, module, grad_input, grad_output, mat):
        d_sigma = module.output * (1. - module.output)
        return super().jac_t_mat_prod(d_sigma, mat)

    def jac_mat_prod(self, module, grad_input, grad_output, mat):
        d_sigma = module.output * (1. - module.output)
        return super().jac_mat_prod(d_sigma, mat)

    def hessian_is_zero(self):
        return False

    def hessian_is_diagonal(self):
        return True

    def hessian_diagonal(self, module, grad_input, grad_output):
        sigma = module.output
        d2_sigma = sigma * (1 - sigma) * (1 - 2 * sigma)
        return super().hessian_diagonal(d2_sigma, grad_output[0])
