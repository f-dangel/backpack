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
        return self.jac_t_mat_prod(module, g_inp, g_out, mat)

    # Transpose Jacobian-matrix product
    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        """
        Note:
        -----
        The Jacobian is *not independent* among the batch dimension, i.e.
        D z_i = D z_i(x_1, ..., x_B).

        This structure breaks the computation of the GGN diagonal,
        for curvature-matrix products it should still work.

        References:
        -----------
        https://kevinzakka.github.io/2016/09/14/batch_normalization/
        https://chrisyeh96.github.io/2017/08/28/deriving-batchnorm-backprop.html
        """
        assert module.affine is True

        batch = self.get_batch(module)
        x_hat, var = self.get_normalized_input_and_var(module)
        ivar = 1. / (var + module.eps).sqrt()

        dx_hat = einsum('bic,i->bic', (mat, module.weight))

        jac_t_mat = batch * dx_hat
        jac_t_mat -= dx_hat.sum(0).unsqueeze(0).expand_as(jac_t_mat)
        jac_t_mat -= einsum('bi,sic,si->bic', (x_hat, dx_hat, x_hat))
        jac_t_mat = einsum('bic,i->bic', (jac_t_mat, ivar / batch))

        return jac_t_mat

    def get_normalized_input_and_var(self, module):
        input = self.get_input(module)
        mean = input.mean(dim=0)
        var = input.var(dim=0, unbiased=False)
        return (input - mean) / (var + module.eps).sqrt(), var

    @jmp_unsqueeze_if_missing_dim(mat_dim=2)
    def weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        batch = self.get_batch(module)
        x_hat, _ = self.get_normalized_input_and_var(module)
        return einsum('bi,ic->bic', (x_hat, mat))

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        x_hat, _ = self.get_normalized_input_and_var(module)
        equation = 'bic,bi->{}ic'.format('' if sum_batch is True else 'b')
        operands = [mat, x_hat]
        return einsum(equation, operands)

    @jmp_unsqueeze_if_missing_dim(mat_dim=2)
    def bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        batch = self.get_batch(module)
        return mat.unsqueeze(0).repeat(batch, 1, 1)

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        if sum_batch is True:
            return mat.sum(0)
        else:
            return mat
