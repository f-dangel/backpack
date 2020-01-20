from backpack.core.derivatives.utils import jac_mat_prod_accept_vectors

from backpack.utils.einsum import einsum
from backpack.core.derivatives.basederivatives import BaseDerivatives


class ElementwiseDerivatives(BaseDerivatives):
    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        df_elementwise = self.df(module, g_inp, g_out)
        return einsum("...,v...->v...", (df_elementwise, mat))

    @jac_mat_prod_accept_vectors
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        return self.jac_t_mat_prod(module, g_inp, g_out, mat)

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        batch, df_flat = self.batch_flat(self.df(module, g_inp, g_out))
        return einsum("ni,nj,ij->ij", (df_flat, df_flat, mat)) / batch

    def hessian_diagonal(self, module, g_inp, g_out):
        return self.d2f(module, g_inp, g_out) * g_out[0]

    def df(self, module, g_inp, g_out):
        raise NotImplementedError("First derivatives not implemented")

    def d2f(self, module, g_inp, g_out):
        raise NotImplementedError("Second derivatives not implemented")
