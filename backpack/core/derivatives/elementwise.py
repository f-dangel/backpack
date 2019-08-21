from ...utils.utils import einsum
from .basederivatives import BaseDerivatives
from .utils import hmp_unsqueeze_if_missing_dim, jmp_unsqueeze_if_missing_dim


class ElementwiseDerivatives(BaseDerivatives):
    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def jac_t_mat_prod(self, module, g_inp, g_out, mat):
        _, df_flat = self.batch_flat(self.df(module, g_inp, g_out))
        return einsum('bi,bic->bic', (df_flat, mat))

    @jmp_unsqueeze_if_missing_dim(mat_dim=3)
    def jac_mat_prod(self, module, g_inp, g_out, mat):
        _, df_flat = self.batch_flat(self.df(module, g_inp, g_out))
        return einsum('bi,bic->bic', (df_flat, mat))

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        batch, df_flat = self.batch_flat(self.df(module, g_inp, g_out))
        return einsum('bi,bj,ij->ij', (df_flat, df_flat, mat)) / batch

    def hessian_diagonal(self, module, g_inp, g_out):
        _, d2f_flat = self.batch_flat(self.d2f(module, g_inp, g_out))
        _, g_out_flat = self.batch_flat(g_out[0])
        return d2f_flat * g_out_flat

    def df(self, module, g_inp, g_out):
        raise NotImplementedError("First derivatives not implemented")

    def d2f(self, module, g_inp, g_out):
        raise NotImplementedError("Second derivatives not implemented")
