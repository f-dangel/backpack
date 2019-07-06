from .context import CTX
from .backpropextension import BackpropExtension


class BackpropSqrtMatrixWithJacobian(BackpropExtension):
    def __init__(self, ctx_name, extension, params=[]):
        super().__init__(self.get_module(), extension, params=params)
        self.__CTX_NAME = ctx_name

    def get_sqrt_matrix(self):
        mat = getattr(CTX, self.__CTX_NAME, None)
        if mat is None:
            raise ValueError(
                "Matrix {} for backpropagation does not exist in CTX".format(
                    self.__CTX_NAME))
        return mat

    def set_sqrt_matrix(self, mat):
        setattr(CTX, self.__CTX_NAME, mat)

    def backpropagate(self, module, grad_input, grad_output):
        sqrt_out = self.get_sqrt_matrix()
        sqrt_in = self.jac_mat_prod(module, grad_input, grad_output, sqrt_out)
        self.set_sqrt_matrix(sqrt_in)
