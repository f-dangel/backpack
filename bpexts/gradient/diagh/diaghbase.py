from ..context import CTX
from ..backpropextension import BackpropExtension
from ..extensions import DIAG_H


class DiagHBase(BackpropExtension):

    def __init__(self, params=[]):
        super().__init__(self.get_module(), DIAG_H, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        for i, sqrt_h_out in enumerate(CTX._backpropagated_sqrt_h):
            sqrt_h_in = self.jac_mat_prod(module, grad_input, grad_output, sqrt_h_out)
            CTX._backpropagated_sqrt_h[i] = sqrt_h_in
