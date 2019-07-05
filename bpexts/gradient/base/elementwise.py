from ...utils import einsum
from ..backpropextension import BackpropExtension


class BaseElementwise(BackpropExtension):

    def jac_mat_prod(self, df, mat):
        return einsum('bi,bic->bic', (df, mat))
