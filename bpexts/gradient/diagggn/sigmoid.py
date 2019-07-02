import torch.nn
from ..config import CTX
from ..jmp.sigmoid import jac_mat_prod
from ..backpropextension import BackpropExtension


class DiagGGNSigmoid(BackpropExtension):

    def __init__(self):
        super().__init__(
            torch.nn.Sigmoid, "DIAG_GGN",
            req_inputs=[0], req_output=True
        )

    def apply(self, module, grad_input, grad_output):
        sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
        sqrt_ggn_in = jac_mat_prod(module, grad_input, grad_output, sqrt_ggn_out)
        CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


SIGNATURE = [(torch.nn.Sigmoid, "DIAG_GGN", DiagGGNSigmoid())]
