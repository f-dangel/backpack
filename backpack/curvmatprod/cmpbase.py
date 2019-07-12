from ..backpropextension import BackpropExtension
from ..context import set_in_ctx, get_from_ctx
from ..extensions import CMP
from ..utils.utils import einsum
from ..curvature import Curvature


class CMPBase(BackpropExtension):
    """Given matrix-vector product routine `MVP(A)` backpropagate
     to `MVP(J^T A J)`."""
    BACKPROPAGATED_CMP_NAME_IN_CTX = "_cmp_backpropagated_mp"
    EXTENSION = CMP

    def __init__(self, params=None):
        if params is None:
            params = []
        BackpropExtension.__init__(
            self, self.get_module(), self.EXTENSION, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        CMP_out = self.get_cmp_from_ctx()

        # second-order module effects
        residual = self._compute_residual_diag_if_nonzero(
            module, grad_input, grad_output)
        residual_mod = Curvature.modify_residual(residual)

        def CMP_in(mat):
            """Multiplication of curvature matrix with matrix `mat`.

            Parameters:
            -----------
            mat : torch.Tensor
                Matrix that will be multiplied.
            """
            Jmat = self.jac_mat_prod(module, grad_input, grad_output, mat)
            CJmat = CMP_out(Jmat)
            JTCJmat = self.jac_t_mat_prod(module, grad_input, grad_output,
                                          CJmat)

            if residual_mod is not None:
                JTCJmat.add_(einsum('bi,bic->bic', (residual_mod, mat)))

            return JTCJmat

        self.set_cmp_in_ctx(CMP_in)

    def _compute_residual_diag_if_nonzero(self, module, grad_input,
                                          grad_output):
        if self.hessian_is_zero():
            return None

        if not self.hessian_is_diagonal():
            raise AttributeError(
                "Residual terms are only supported for elementwise functions")

        # second order module effects
        return self.hessian_diagonal(module, grad_input, grad_output)

    def get_cmp_from_ctx(self):
        return get_from_ctx(self.BACKPROPAGATED_CMP_NAME_IN_CTX)

    def set_cmp_in_ctx(self, value):
        set_in_ctx(self.BACKPROPAGATED_CMP_NAME_IN_CTX, value)
