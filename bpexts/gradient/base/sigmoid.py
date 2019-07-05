from torch.nn import Sigmoid
from .elementwise import BaseElementwise


class BaseSigmoid(BaseElementwise):

    def get_module(self):
        return Sigmoid

    def jac_mat_prod(self, module, grad_input, grad_output, mat):
        d_sigma = module.output * (1. - module.output)
        return super().jac_mat_prod(d_sigma, mat)
