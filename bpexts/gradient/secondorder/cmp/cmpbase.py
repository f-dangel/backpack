from ...backpropextension import BackpropExtension
from ...ctxinteract import ActOnCTX
from ...extensions import CMP
from ....utils import einsum
from ..curvature import Curvature

BACKPROPAGATED_MP_NAME = "_cmp_backpropagated_mp"
EXTENSION = CMP


class CMPBase(BackpropExtension, ActOnCTX):
    """Given matrix-vector product routine `MVP(A)` backpropagate
     to `MVP(J^T A J)`."""

    def __init__(self, params=None):
        if params is None:
            params = []
        ActOnCTX.__init__(self, BACKPROPAGATED_MP_NAME)
        BackpropExtension.__init__(
            self, self.get_module(), EXTENSION, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        CMP_out = self.get_from_ctx()

        # second-order module effects
        residual = self._compute_residual_diag_if_nonzero(
            module, grad_input, grad_output)

        def CMP_in(mat, which):
            """Multiplication of curvature matrix with matrix `mat`.

            Parameters:
            -----------
            mat : torch.Tensor
                Matrix that will be multiplied.
            which : str
                Which curvature matrix to use. For choices,
                Choices: See `CURVATURE_CHOICES`
            """
            Jmat = self.jac_mat_prod(module, grad_input, grad_output, mat)
            CJmat = CMP_out(Jmat, which=which)
            JTCJmat = self.jac_t_mat_prod(module, grad_input, grad_output,
                                          CJmat)

            res_mod = Curvature.modify_residual(residual, which)
            if res_mod is not None:
                JTCJmat.add_(einsum('bi,bic->bic', (res_mod, mat)))

            return JTCJmat

        self.set_in_ctx(CMP_in)

    def _compute_residual_diag_if_nonzero(self, module, grad_input,
                                          grad_output):
        if self.hessian_is_zero():
            return None

        if not self.hessian_is_diagonal():
            raise AttributeError(
                "Residual terms are only supported for elementwise functions")

        # second order module effects
        return self.hessian_diagonal(module, grad_input, grad_output)
