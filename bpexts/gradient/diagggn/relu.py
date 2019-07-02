import torch.nn
from ..context import CTX
from ..jmp.relu import jac_mat_prod
from ..backpropextension import BackpropExtension
from ..extensions import DIAG_GGN


class DiagGGNReLU(BackpropExtension):

    def __init__(self):
        super().__init__(
            torch.nn.ReLU, DIAG_GGN,
            req_inputs=[0], req_output=True
        )

    def apply(self, module, grad_input, grad_output):
        sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
        sqrt_ggn_in = jac_mat_prod(module, grad_input, grad_output, sqrt_ggn_out)
        CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


EXTENSIONS = [DiagGGNReLU()]
