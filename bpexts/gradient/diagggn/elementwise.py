from ..context import CTX
from ..backpropextension import BackpropExtension
from ..extensions import DIAG_GGN


class DiagGGNElementwise(BackpropExtension):

    def __init__(self):
        super().__init__(
            self.get_module(), DIAG_GGN,
        )

    def apply(self, module, grad_input, grad_output):
        sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
        sqrt_ggn_in = self.jac_mat_prod(module, grad_input, grad_output, sqrt_ggn_out)
        CTX._backpropagated_sqrt_ggn = sqrt_ggn_in
