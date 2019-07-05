from ...context import CTX
from ...backpropextension import BackpropExtension
from ...extensions import DIAG_GGN


class DiagGGNBase(BackpropExtension):

    def __init__(self, params=[]):
        super().__init__(self.get_module(), DIAG_GGN, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
        sqrt_ggn_in = self.jac_mat_prod(module, grad_input, grad_output, sqrt_ggn_out)
        CTX._backpropagated_sqrt_ggn = sqrt_ggn_in
