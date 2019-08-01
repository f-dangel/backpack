import torch
import torch.nn
from torch.nn import BatchNorm1d

from ...utils.utils import einsum
from .basederivatives import BaseDerivatives
from .utils import jmp_unsqueeze_if_missing_dim


class BatchNorm1dDerivatives(BaseDerivatives):
    def get_module(self):
        return BatchNorm1d

    def hessian_is_zero(self):
        return False

    def hessian_is_diagonal(self):
        return False

    # Jacobian-matrix product
    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        raise NotImplementedError

    # Transpose Jacobian-matrix product
    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        # Need tests
        raise NotImplementedError
        # assert module.affine is True
        # batch = self.get_batch(module)
        # x_hat, var = self.get_normalized_input_and_var(module)

        # x_hat_mat = mat * module.weight

        # jac_t_mat = batch * x_hat_mat
        # jac_t_mat -= x_hat_mat.sum(0)
        # jac_t_mat -= x_hat * (x_hat_mat * x_hat).sum(0)
        # jac_t_mat /= (var + module.eps).sqrt() * batch

        # return jac_t_mat

    def get_normalized_input_and_var(self, module):
        input = self.get_input(module)
        mean = input.mean(dim=0)
        var = input.var(dim=0, unbiased=False)
        # var2 = ((input - input.mean(0))**2).mean(0)
        # assert torch.allclose(var, var2)
        return (input - mean) / (var + module.eps).sqrt(), var

    @jmp_unsqueeze_if_missing_dim(mat_dim=2)
    def weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        raise NotImplementedError

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        x_hat, _ = self.get_normalized_input_and_var(module)
        equation = 'bic,bi->{}ic'.format('' if sum_batch is True else 'b')
        operands = [mat, x_hat]
        return einsum(equation, operands)

    @jmp_unsqueeze_if_missing_dim(mat_dim=2)
    def bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        raise NotImplementedError

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        if sum_batch is True:
            return mat.sum(0)
        else:
            return mat
