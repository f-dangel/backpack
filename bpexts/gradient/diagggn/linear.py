import torch.nn
from ..config import CTX
from ..jmp.linear import jac_mat_prod
from ...utils import einsum
from ..backpropextension import BackpropExtension


class DiagGGNLinear(BackpropExtension):

    def __init__(self):
        super().__init__(
            torch.nn.Linear, "DIAG_GGN",
            req_inputs=[0]
        )

    def apply(self, module, grad_input, grad_output):
        sqrt_ggn_out = CTX._backpropagated_sqrt_ggn

        if module.bias is not None and module.bias.requires_grad:
            module.bias.diag_ggn = self.bias_diag_ggn(module, grad_output, sqrt_ggn_out)
        if module.weight.requires_grad:
            module.weight.diag_ggn = self.weight_diag_ggn(module, grad_output,
                                                          sqrt_ggn_out)

        self.backpropagate_sqrt_ggn(module, grad_input, grad_output, sqrt_ggn_out)

    def bias_diag_ggn(self, module, grad_output, sqrt_ggn_out):
        sqrt_ggn_bias = sqrt_ggn_out
        return einsum('bic->i', (sqrt_ggn_bias**2, ))

    def weight_diag_ggn(self, module, grad_output, sqrt_ggn_out):
        return einsum('bic,bj->ij', (sqrt_ggn_out**2, module.input0**2))

    def backpropagate_sqrt_ggn(self, module, grad_input, grad_output, sqrt_ggn_out):
        sqrt_ggn_in = jac_mat_prod(module, grad_input, grad_output, sqrt_ggn_out)
        CTX._backpropagated_sqrt_ggn = sqrt_ggn_in


SIGNATURE = [(torch.nn.Linear, "DIAG_GGN", DiagGGNLinear())]
