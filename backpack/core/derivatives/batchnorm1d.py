from warnings import warn

from torch import einsum
from torch.nn import BatchNorm1d

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives
from backpack.core.derivatives.shape_check import (
    R_mat_prod_accept_vectors,
    R_mat_prod_check_shapes,
)


class BatchNorm1dDerivatives(BaseParameterDerivatives):
    def get_module(self):
        return BatchNorm1d

    def hessian_is_zero(self):
        return False

    def hessian_is_diagonal(self):
        return False

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        return self._jac_t_mat_prod(module, g_inp, g_out, mat)

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
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

        N = self.get_batch(module)
        x_hat, var = self.get_normalized_input_and_var(module)
        ivar = 1.0 / (var + module.eps).sqrt()

        dx_hat = einsum("vni,i->vni", (mat, module.weight))

        jac_t_mat = N * dx_hat
        jac_t_mat -= dx_hat.sum(1).unsqueeze(1).expand_as(jac_t_mat)
        jac_t_mat -= einsum("ni,vsi,si->vni", (x_hat, dx_hat, x_hat))
        jac_t_mat = einsum("vni,i->vni", (jac_t_mat, ivar / N))

        return jac_t_mat

    def get_normalized_input_and_var(self, module):
        input = module.input0
        mean = input.mean(dim=0)
        var = input.var(dim=0, unbiased=False)
        return (input - mean) / (var + module.eps).sqrt(), var

    @R_mat_prod_accept_vectors
    @R_mat_prod_check_shapes
    def make_residual_mat_prod(self, module, g_inp, g_out):
        # TODO: Implement R_mat_prod for BatchNorm
        def R_mat_prod(mat):
            """Multiply with the residual: mat → [∑_{k} Hz_k(x) 𝛿z_k] mat.

            Second term of the module input Hessian backpropagation equation.
            """
            raise NotImplementedError

        # TODO: Enable tests in test/automated_bn_test.py
        raise NotImplementedError
        return R_mat_prod

    def _weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        x_hat, _ = self.get_normalized_input_and_var(module)
        return einsum("ni,vi->vni", (x_hat, mat))

    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch):
        if not sum_batch:
            warn(
                "BatchNorm batch summation disabled."
                "This may not compute meaningful quantities"
            )
        x_hat, _ = self.get_normalized_input_and_var(module)
        equation = "vni,ni->v{}i".format("" if sum_batch is True else "n")
        operands = [mat, x_hat]
        return einsum(equation, operands)

    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        N = self.get_batch(module)
        return mat.unsqueeze(1).repeat(1, N, 1)

    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        if not sum_batch:
            warn(
                "BatchNorm batch summation disabled."
                "This may not compute meaningful quantities"
            )
            return mat
        else:
            N_axis = 1
            return mat.sum(N_axis)
