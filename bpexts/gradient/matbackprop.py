from .context import CTX
from .backpropextension import BackpropExtension


class ActOnCTX():
    def __init__(self, ctx_name):
        self.__CTX_NAME = ctx_name

    def get_from_ctx(self):
        value = getattr(CTX, self.__CTX_NAME, None)
        if value is None:
            raise ValueError(
                "Attribute {} for backpropagation does not exist in CTX".
                format(self.__CTX_NAME))
        return value

    def set_in_ctx(self, value):
        setattr(CTX, self.__CTX_NAME, value)


class MatToJacMat(BackpropExtension, ActOnCTX):
    """Backpropagate `M` to `J^T M`."""

    def __init__(self, ctx_name, extension, params=[]):
        ActOnCTX.__init__(self, ctx_name)
        BackpropExtension.__init__(
            self, self.get_module(), extension, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        sqrt_out = self.get_from_ctx()
        sqrt_in = self.jac_mat_prod(module, grad_input, grad_output, sqrt_out)
        self.set_in_ctx(sqrt_in)
