from backpack.extensions.mat_to_mat_jac_base import MatToJacMat
from torch import clamp, diag_embed


class DiagHBaseModule(MatToJacMat):
    PLUS = 1.0
    MINUS = -1.0

    def __init__(self, derivatives, params=None):
        super().__init__(derivatives=derivatives, params=params)

    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        bp_matrices = backproped["matrices"]
        bp_signs = backproped["signs"]

        bp_matrices = super().backpropagate(ext, module, g_inp, g_out, bp_matrices)

        for matrix, sign in self.__local_curvatures(module, g_inp, g_out):
            bp_matrices.append(matrix)
            bp_signs.append(sign)

        return {"matrices": bp_matrices, "signs": bp_signs}

    def __local_curvatures(self, module, g_inp, g_out):
        if self.derivatives.hessian_is_zero():
            return []
        if not self.derivatives.hessian_is_diagonal():
            raise NotImplementedError

        def positive_part(sign, H):
            return clamp(sign * H, min=0)

        def decompose_into_positive_and_negative_sqrt(H):
            return [
                [diag_embed(positive_part(sign, H).sqrt_()), sign]
                for sign in [self.PLUS, self.MINUS]
            ]

        H = self.derivatives.hessian_diagonal(module, g_inp, g_out)
        return decompose_into_positive_and_negative_sqrt(H)
