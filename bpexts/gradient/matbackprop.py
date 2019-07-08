from .backpropextension import BackpropExtension
from .ctxinteract import ActOnCTX


class MatToJacMat(BackpropExtension, ActOnCTX):
    """Backpropagate `M` to `J^T M`."""

    def __init__(self, ctx_name, extension, params=None):
        if params is None:
            params = []
        ActOnCTX.__init__(self, ctx_name)
        BackpropExtension.__init__(
            self, self.get_module(), extension, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        sqrt_out = self.get_from_ctx()
        sqrt_in = self.jac_mat_prod(module, grad_input, grad_output, sqrt_out)
        self.set_in_ctx(sqrt_in)
