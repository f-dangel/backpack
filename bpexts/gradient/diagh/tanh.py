import torch.nn
from ..context import CTX
from ..backpropextension import BackpropExtension
from ..jmp.tanh import jac_mat_prod
from ..extensions import DIAG_H

DETACH_INPUTS = True


class DiagHTanh(BackpropExtension):
    def __init__(self):
        super().__init__(torch.nn.Tanh, DIAG_H, req_inputs=[0])

    def apply(self, module, grad_input, grad_output):
        sqrt_h_outs = CTX._backpropagated_sqrt_h
        sqrt_h_outs_signs = CTX._backpropagated_sqrt_h_signs

        if module.input0.requires_grad or DETACH_INPUTS:
            self.backpropagate_sqrt_h(module, grad_input, grad_output,
                                      sqrt_h_outs, sqrt_h_outs_signs)

    def backpropagate_sqrt_h(self, module, grad_input, grad_output,
                             sqrt_h_outs, sqrt_h_outs_signs):
        for i, sqrt_h in enumerate(sqrt_h_outs):
            sqrt_h_outs[i] = jac_mat_prod(module, grad_input, grad_output,
                                          sqrt_h)
        # spawn backpropagation of residual sqrt matrices
        self.add_residuals_to_backprop(module, grad_output, sqrt_h_outs,
                                       sqrt_h_outs_signs)

    def add_residuals_to_backprop(self, module, grad_output, sqrt_h_outs,
                                  sqrt_h_outs_signs):
        for sqrt_h, sign in self.sqrt_h_residuals(module, grad_output):
            sqrt_h_outs.append(sqrt_h)
            sqrt_h_outs_signs.append(sign)

    def sqrt_h_residuals(self, module, grad_output):
        batch, tanh = grad_output[0].size(0), module.output
        d2_tanh = (-2. * tanh * (1. - tanh**2))

        res = d2_tanh.view(batch, -1) * grad_output[0].view(batch, -1)

        # diagonals
        sqrt_res_plus = torch.clamp(res, min=0, max=float('inf')).sqrt_()
        sqrt_res_minus = torch.clamp(-res, min=0, max=float('inf')).sqrt_()

        # matrices (batch, features, 1)
        sqrt_res_plus_mat = torch.diag_embed(sqrt_res_plus)
        sqrt_res_minus_mat = torch.diag_embed(sqrt_res_minus)

        return [(sqrt_res_plus_mat, 1.), (sqrt_res_minus_mat, -1.)]


EXTENSIONS = [DiagHTanh()]
