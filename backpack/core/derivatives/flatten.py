from torch.nn import Flatten

from backpack.core.derivatives.utils import jac_t_new_shape_convention
from backpack.utils.unsqueeze import jmp_unsqueeze_if_missing_dim

from .basederivatives import BaseDerivatives


class FlattenDerivatives(BaseDerivatives):
    def get_module(self):
        return Flatten

    def hessian_is_zero(self):
        return True

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        return mat

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    @jac_t_new_shape_convention
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        new_convention = True
        if new_convention:
            return mat
        else:
            return mat

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        return mat
