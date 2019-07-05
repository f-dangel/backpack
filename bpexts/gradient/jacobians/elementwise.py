from .basejacobian import BaseJacobian
from ...utils import einsum


class ElementwiseJacobian(BaseJacobian):

    def jac_mat_prod(self, df, mat):
        return einsum('bi,bic->bic', (df, mat))
