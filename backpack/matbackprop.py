from .backpropextension import BackpropExtension
from .context import get_from_ctx, set_in_ctx


class MatToJacMat(BackpropExtension):
    """Backpropagate `M` to `J^T M`."""

    def __init__(self, mat_name_in_ctx, extension, params=None):
        if params is None:
            params = []
        super().__init__(self.get_module(), extension, params=params)
        self.MAT_NAME_IN_CTX = mat_name_in_ctx

    def backpropagate(self, module, grad_input, grad_output):
        M = self.get_mat_from_ctx()

        if isinstance(M, list):
            JT_M = self.apply_jac_t_on_list(module, grad_input, grad_output, M)
        else:
            JT_M = self.jac_t_mat_prod(module, grad_input, grad_output, M)

        self.set_mat_in_ctx(JT_M)

    def apply_jac_t(self, module, grad_input, grad_output, M):
        return self.jac_t_mat_prod(module, grad_input, grad_output, M)

    def apply_jac_t_on_list(self, module, grad_input, grad_output, M_list):
        M_list = [
            self.apply_jac_t(module, grad_input, grad_output, M)
            for M in M_list
        ]
        return M_list

    def get_mat_from_ctx(self):
        return get_from_ctx(self.MAT_NAME_IN_CTX)

    def set_mat_in_ctx(self, mat):
        set_in_ctx(self.MAT_NAME_IN_CTX, mat)
