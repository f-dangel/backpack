from torch.nn import Flatten

from backpack.core.derivatives.utils import (
    jac_t_mat_prod_accept_vectors,
    jac_mat_prod_accept_vectors,
)

from .basederivatives import BaseDerivatives


class FlattenDerivatives(BaseDerivatives):
    def get_module(self):
        return Flatten

    def hessian_is_zero(self):
        return True

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        return mat

    @jac_t_mat_prod_accept_vectors
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        return self.view_like_input(mat, module)

    @jac_mat_prod_accept_vectors
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        return self.view_like_output(mat, module)
