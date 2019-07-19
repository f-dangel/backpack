from ..backpropextension import BackpropExtension
from ..context import set_in_ctx, get_from_ctx
from ..extensions import CMP
from ..utils.utils import einsum
from ..curvature import Curvature
from ..core.derivatives.utils import hmp_unsqueeze_if_missing_dim


class CMPBase(BackpropExtension):
    """Given matrix-vector product routine `MVP(A)` backpropagate
     to `MVP(J^T A J)`."""
    BACKPROPAGATED_CMP_NAME_IN_CTX = "_cmp_backpropagated_mp"
    EXTENSION = CMP

    def __init__(self, params=None):
        if params is None:
            params = []
        super().__init__(self.get_module(), self.EXTENSION, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        CMP_out = self.get_cmp_from_ctx()

        # second-order module effects
        residual = self._compute_residual_diag_if_nonzero(
            module, grad_input, grad_output)
        residual_mod = self._modify_residual(residual)

        @hmp_unsqueeze_if_missing_dim(mat_dim=3)
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

    def _modify_residual(self, residual):
        curv_type = self._get_curv_type_from_extension()
        return Curvature.modify_residual(residual, curv_type)

    def _get_curv_type_from_extension(self):
        return self._get_parametrized_ext().get_curv_type()

    def get_cmp_from_ctx(self):
        return get_from_ctx(self.BACKPROPAGATED_CMP_NAME_IN_CTX)

    def set_cmp_in_ctx(self, value):
        set_in_ctx(self.BACKPROPAGATED_CMP_NAME_IN_CTX, value)
