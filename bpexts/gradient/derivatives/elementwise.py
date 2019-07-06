from .basederivatives import BaseDerivatives
from ...utils import einsum


class ElementwiseDerivatives(BaseDerivatives):

    def jac_mat_prod(self, df, mat):
        return einsum('bi,bic->bic', (df, mat))

    def hessian_diagonal(self, ddf, mat):
        batch = mat.shape[0]
        return ddf.view(batch, -1) * mat.view(batch, -1)
