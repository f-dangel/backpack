from backpack.extensions.mat_to_mat_jac_base import MatToJacMat
from torch import clamp, diag_embed


class DiagHBaseModule(MatToJacMat):
    PLUS = 1.
    MINUS = -1.

    def __init__(self, derivatives, params=None):
        super().__init__(derivatives=derivatives, params=params)

    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        bp_matrices = backproped["matrices"]
        bp_signs = backproped["signs"]

        bp_matrices = super().backpropagate(ext, module, g_inp, g_out, bp_matrices)

        for matrix, sign in self.local_curvatures(module, g_inp, g_out):
            bp_matrices.append(matrix)
            bp_signs.append(sign)

        return {
            "matrices": bp_matrices,
            "signs": bp_signs
        }

    def local_curvatures(self, module, g_inp, g_out):
        diag_h_sign = []

        if not self.derivatives.hessian_is_zero():

            def positive_part(sign, H):
                return clamp(sign * H, min=0, max=float("inf"))

            def decompose_into_positive_and_negative_sqrt(H):
                return [
                    [diag_embed(positive_part(sign, H).sqrt_()), sign]
                    for sign in [self.PLUS, self.MINUS]
                ]

            if not self.derivatives.hessian_is_diagonal():
                raise NotImplementedError

            H = self.derivatives.hessian_diagonal(module, g_inp, g_out)
            diag_h_sign += decompose_into_positive_and_negative_sqrt(H)

        return diag_h_sign
