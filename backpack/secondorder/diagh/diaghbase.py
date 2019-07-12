from torch import clamp, diag_embed
from ...context import get_from_ctx, set_in_ctx
from ...extensions import DIAG_H
from ...matbackprop import MatToJacMat


class DiagHBase(MatToJacMat):
    MAT_LIST_NAME_IN_CTX = "_backpropagated_sqrt_h"
    SIGN_LIST_NAME_IN_CTX = "_backpropagated_sqrt_h_signs"
    EXTENSION = DIAG_H
    PLUS = 1.
    MINUS = -1.

    def __init__(self, params=None):
        if params is None:
            params = []
        super().__init__(
            self.MAT_LIST_NAME_IN_CTX, self.EXTENSION, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        super().backpropagate(module, grad_input, grad_output)
        self.append_residuals_and_signs_to_lists(module, grad_input,
                                                 grad_output)

    def append_residuals_and_signs_to_lists(self, module, grad_input,
                                            grad_output):
        if self.hessian_is_zero():
            return
        if not self.hessian_is_diagonal():
            raise NotImplementedError

        H = self.hessian_diagonal(module, grad_input, grad_output)

        for sign in [self.PLUS, self.MINUS]:
            Hsign = clamp(sign * H, min=0, max=float('inf')).sqrt_()
            self.get_mat_from_ctx().append(diag_embed(Hsign))
            self.get_sign_list_from_ctx().append(sign)

    def get_sign_list_from_ctx(self):
        return get_from_ctx(self.SIGN_LIST_NAME_IN_CTX)

    def set_sign_list_in_ctx(self, sign_list):
        set_in_ctx(self.SIGN_LIST_NAME_IN_CTX, sign_list)
