from .basederivatives import BaseDerivatives
from ...utils import einsum


class ElementwiseDerivatives(BaseDerivatives):
    def jac_t_mat_prod(self, df, mat):
        batch = df.size(0)
        return einsum('bi,bic->bic', (df.view(batch, -1), mat))

    def hessian_diagonal(self, ddf, mat):
        batch = mat.shape[0]
        return ddf.view(batch, -1) * mat.view(batch, -1)
