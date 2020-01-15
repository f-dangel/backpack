import torch
from torch.nn import Flatten

from backpack.core.derivatives.utils import (
    jac_new_shape_convention,
    jac_t_new_shape_convention,
)
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
            num_cols = mat.shape[0]
            shape = (num_cols,) + tuple(module.input0_shape)
        else:
            batch = module.get_batch()
            features = torch.prod(module.input0_shape[1:])
            num_cols = mat.shape[-1]
            shape = (batch, features, num_cols)

        return mat.view(shape)

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    @jac_new_shape_convention
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        new_convention = True

        if new_convention:
            num_cols = mat.shape[0]
            shape = (num_cols,) + tuple(module.output_shape)
        else:
            batch = module.get_batch()
            features = torch.prod(module.output_shape[1:])
            num_cols = mat.shape[-1]
            shape = (batch, features, num_cols)

        return mat.view(shape)
