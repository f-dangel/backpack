from ...context import CTX
from ...backpropextension import BackpropExtension
from ...extensions import KFLR


class KFLRBase(BackpropExtension):
    def __init__(self, params=[]):
        super().__init__(self.get_module(), KFLR, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        kflr_sqrt_ggn_out = CTX._kflr_backpropagated_sqrt_ggn
        kflr_sqrt_ggn_in = self.jac_mat_prod(module, grad_input, grad_output,
                                             kflr_sqrt_ggn_out)
        CTX._kflr_backpropagated_sqrt_ggn = kflr_sqrt_ggn_in
