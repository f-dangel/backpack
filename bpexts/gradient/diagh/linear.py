import torch
import torch.nn
from ..context import CTX
from ...utils import einsum
from ..backpropextension import BackpropExtension
from ..jmp.linear import jac_mat_prod
from ..extensions import DIAG_H


class DiagHLinear(BackpropExtension):
    def __init__(self):
        super().__init__(torch.nn.Linear, DIAG_H, req_inputs=[0])

    def apply(self, module, grad_input, grad_output):
        sqrt_h_outs = CTX._backpropagated_sqrt_h
        sqrt_h_outs_signs = CTX._backpropagated_sqrt_h_signs

        if module.bias is not None and module.bias.requires_grad:
            module.bias.diag_h = self.bias_diagH(module, sqrt_h_outs,
                                                 sqrt_h_outs_signs)
        if module.weight.requires_grad:
            module.weight.diag_h = self.weight_diagH(module, sqrt_h_outs,
                                                     sqrt_h_outs_signs)
        if module.input0.requires_grad:
            self.backpropagate_sqrt_h(module, grad_input, grad_output, sqrt_h_outs,
                                      sqrt_h_outs_signs)

    # TODO: Reuse code in ..diaggn.linear to extract the diagonal
    def bias_diagH(self, module, sqrt_h_outs, sqrt_h_outs_signs):
        h_diag = torch.zeros_like(module.bias)
        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(sign * einsum('bic->i', (h_sqrt**2, )))
        return h_diag

    # TODO: Reuse code in ..diaggn.linear to extract the diagonal
    def weight_diagH(self, module, sqrt_h_outs, sqrt_h_outs_signs):
        h_diag = torch.zeros_like(module.weight)
        for h_sqrt, sign in zip(sqrt_h_outs, sqrt_h_outs_signs):
            h_diag.add_(sign * einsum('bic->i', (h_sqrt**2, )))
        return h_diag

    def backpropagate_sqrt_h(self, module, grad_input, grad_output,
                             sqrt_h_outs, sqrt_h_outs_signs):
        for i, sqrt_h in enumerate(sqrt_h_outs):
            sqrt_h_outs[i] = jac_mat_prod(module, grad_input, grad_output,
                                          sqrt_h)
        CTX._backpropagated_sqrt_h = sqrt_h_outs


EXTENSIONS = [DiagHLinear()]
