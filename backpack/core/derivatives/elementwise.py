from .basederivatives import BaseDerivatives
from ...utils.utils import einsum


class ElementwiseDerivatives(BaseDerivatives):
    def jac_t_mat_prod(self, df, mat):
        _, df_flat = self.batch_flat(df)
        return einsum('bi,bic->bic', (df_flat, mat))

    def jac_mat_prod(self, df, mat):
        _, df_flat = self.batch_flat(df)
        return einsum('bi,bic->bic', (df_flat, mat))

    def hessian_diagonal(self, ddf, mat):
        _, ddf_flat = self.batch_flat(ddf)
        _, mat_flat = self.batch_flat(mat)
        return ddf_flat * mat_flat

    def ea_jac_t_mat_jac(self, df, mat):
        batch, df_flat = self.batch_flat(df)
        return einsum('bi,ij,bj->ij', (df_flat, mat, df_flat)) / batch

    @staticmethod
    def batch_flat(tensor):
        batch = tensor.size(0)
        return batch, tensor.view(batch, -1)
