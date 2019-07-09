from ...backpropextension import BackpropExtension
from ...ctxinteract import ActOnCTX
from ...extensions import CMP
from ....utils import einsum

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
        D = self._compute_residual_diag_if_nonzero(module, grad_input,
                                                   grad_output)

        def CMP_in(mat):
            Jmat = self.jac_mat_prod(module, grad_input, grad_output, mat)
            CJmat = CMP_out(Jmat)
            JTCJmat = self.jac_t_mat_prod(module, grad_input, grad_output,
                                          CJmat)

            if D is not None:
                JTCJmat.add_(einsum('bi,bic->bic', (D, mat)))

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
