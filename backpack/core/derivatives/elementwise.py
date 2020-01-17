from backpack.core.derivatives.utils import (
    jac_t_mat_prod_accept_vectors,
    jac_mat_prod_accept_vectors,
)

from ...utils.einsum import einsum
from .basederivatives import BaseDerivatives


class ElementwiseDerivatives(BaseDerivatives):
    @jac_t_mat_prod_accept_vectors
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        new_convention = True

        if new_convention:
            df_non_flat = self.df(module, g_inp, g_out)
            return einsum("...,c...->c...", (df_non_flat, mat))
        else:
            _, df_flat = self.batch_flat(self.df(module, g_inp, g_out))
            return einsum("bi,bic->bic", (df_flat, mat))

    @jac_mat_prod_accept_vectors
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        return self.jac_t_mat_prod(module, g_inp, g_out, mat)

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        batch, df_flat = self.batch_flat(self.df(module, g_inp, g_out))
        return einsum("bi,bj,ij->ij", (df_flat, df_flat, mat)) / batch

    def hessian_diagonal(self, module, g_inp, g_out):
        new_convention = True

        if new_convention:
            return self.d2f(module, g_inp, g_out) * g_out[0]
        else:
            _, d2f_flat = self.batch_flat(self.d2f(module, g_inp, g_out))
            _, g_out_flat = self.batch_flat(g_out[0])
            return d2f_flat * g_out_flat

    def df(self, module, g_inp, g_out):
        raise NotImplementedError("First derivatives not implemented")

    def d2f(self, module, g_inp, g_out):
        raise NotImplementedError("Second derivatives not implemented")
