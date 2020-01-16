from torch.nn import Linear

from backpack.core.derivatives.utils import (
    jac_new_shape_convention,
    jac_t_new_shape_convention,
    weight_jac_t_new_shape_convention,
    bias_jac_t_new_shape_convention,
)
from backpack.utils.unsqueeze import jmp_unsqueeze_if_missing_dim

from ...utils.einsum import einsum
from .basederivatives import BaseDerivatives


class LinearDerivatives(BaseDerivatives):
    def get_module(self):
        return Linear

    def get_input(self, module):
        return module.input0

    def hessian_is_zero(self):
        return True

    def get_weight_data(self, module):
        return module.weight.data

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    @jac_t_new_shape_convention
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        new_convention = True

        d_linear = self.get_weight_data(module)

        if new_convention:
            return einsum("ij,cbi->cbj", (d_linear, mat))
        else:
            return einsum("ij,bic->bjc", (d_linear, mat))

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    @jac_new_shape_convention
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        new_convention = True

        d_linear = self.get_weight_data(module)

        if new_convention:
            return einsum("ij,cbj->cbi", (d_linear, mat))
        else:
            return einsum("ij,bjc->bic", (d_linear, mat))

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        jac = self.get_weight_data(module)
        return einsum("ik,ij,jl->kl", (jac, mat, jac))

    @jmp_unsqueeze_if_missing_dim(mat_dim=2)
    def weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        num_cols = mat.size(1)
        shape = tuple(module.weight.size()) + (num_cols,)

        jac_mat = einsum("bj,ijc->bic", (self.get_input(module), mat.view(shape)))
        return jac_mat

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    @weight_jac_t_new_shape_convention
    def weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        new_convention = True

        batch = self.get_batch(module)

        if new_convention:
            num_cols = mat.size(0)
            equation = "cbj,bi->cji" if sum_batch is True else "cbj,bi->cbji"
        else:
            num_cols = mat.size(2)
            equation = "bjc,bi->jic" if sum_batch is True else "bjc,bi->bjic"

        jac_t_mat = einsum(equation, (mat, self.get_input(module))).contiguous()

        if new_convention:
            if sum_batch:
                shape = (num_cols,) + tuple(module.weight.shape)
            else:
                shape = (num_cols, batch) + tuple(module.weight.shape)
        else:
            sum_shape = [module.weight.numel(), num_cols]
            shape = sum_shape if sum_batch is True else [batch] + sum_shape

        return jac_t_mat.view(shape)

    @jmp_unsqueeze_if_missing_dim(mat_dim=2)
    def bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        batch = self.get_batch(module)
        return mat.unsqueeze(0).expand(batch, -1, -1)

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    @bias_jac_t_new_shape_convention
    def bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        new_convention = True

        if sum_batch is True:
            if new_convention:
                return mat.sum(1)
            else:
                return mat.sum(0)
        else:
            return mat
