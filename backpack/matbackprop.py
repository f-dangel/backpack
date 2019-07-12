from .backpropextension import BackpropExtension
from .ctxinteract import ActOnCTX
from .context import get_from_ctx, set_in_ctx


class MatToJacMat(BackpropExtension, ActOnCTX):
    """Backpropagate `M` to `J^T M`."""

    def __init__(self, mat_name_in_ctx, extension, params=None):
        if params is None:
            params = []
        self.M_NAME_IN_CTX = mat_name_in_ctx
        ActOnCTX.__init__(self, mat_name_in_ctx)
        BackpropExtension.__init__(
            self, self.get_module(), extension, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        M = self.get_M_from_ctx()

        if isinstance(M, list):
            JT_M = self.apply_jac_t_on_list(module, grad_input, grad_output, M)
        else:
            JT_M = self.jac_t_mat_prod(module, grad_input, grad_output, M)

        self.set_JT_M_in_ctx(JT_M)

    def apply_jac_t(self, module, grad_input, grad_output, M):
        return self.jac_t_mat_prod(module, grad_input, grad_output, M)

    def apply_jac_t_on_list(self, module, grad_input, grad_output, M_list):
        mat_list = [
            self.apply_jac_t(module, grad_input, grad_output, M)
            for M in M_list
        ]
        return M_list

    def get_M_from_ctx(self):
        return get_from_ctx(self.M_NAME_IN_CTX)

    def set_JT_M_in_ctx(self, JT_M):
        set_in_ctx(self.M_NAME_IN_CTX, JT_M)
