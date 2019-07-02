import torch.nn
from ..config import CTX
from ..jmp.dropout import jac_mat_prod
from ..backpropextension import BackpropExtension


class DiagGGNDropout(BackpropExtension):

    def __init__(self):
        super().__init__(req_inputs=[0], req_output=True)

    def apply(self, module, grad_input, grad_output):
        sqrt_ggn_out = CTX._backpropagated_sqrt_ggn
        sqrt_ggn_in = jac_mat_prod(module, grad_input, grad_output, sqrt_ggn_out)
        CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


SIGNATURE = [(torch.nn.Dropout, "DIAG_GGN", DiagGGNDropout())]
