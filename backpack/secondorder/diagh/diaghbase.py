from torch import clamp, diag_embed
from ...context import CTX
from ...backpropextension import BackpropExtension
from ...extensions import DIAG_H


class DiagHBase(BackpropExtension):
    def __init__(self, params=[]):
        super().__init__(self.get_module(), DIAG_H, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        for i, sqrt_h_out in enumerate(CTX._backpropagated_sqrt_h):
            sqrt_h_in = self.jac_t_mat_prod(module, grad_input, grad_output,
                                            sqrt_h_out)
            CTX._backpropagated_sqrt_h[i] = sqrt_h_in

        if not self.hessian_is_zero():
            if self.hessian_is_diagonal():
                H = self.hessian_diagonal(module, grad_input, grad_output)
                Hplus = clamp(H, min=0, max=float('inf')).sqrt_()
                Hminus = clamp(-H, min=0, max=float('inf')).sqrt_()

                CTX._backpropagated_sqrt_h.append(diag_embed(Hplus))
                CTX._backpropagated_sqrt_h_signs.append(1.)

                CTX._backpropagated_sqrt_h.append(diag_embed(Hminus))
                CTX._backpropagated_sqrt_h_signs.append(-1.)
            else:
                raise NotImplementedError
