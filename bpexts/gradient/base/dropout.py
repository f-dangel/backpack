from torch import eq
from torch.nn import Dropout
from .elementwise import BaseElementwise


class BaseDropout(BaseElementwise):

    def get_module(self):
        return Dropout

    def jac_mat_prod(self, module, grad_input, grad_output, mat):
        scaling = 1 / (1 - module.p)
        mask = 1 - eq(grad_input, 0.).float()
        d_dropout = mask * scaling
        return super().jac_mat_prod(d_dropout, mat)
