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
        if self.derivatives is None or self.derivatives.hessian_is_zero():
            return []
        if not self.derivatives.hessian_is_diagonal():
            raise NotImplementedError

        H = self.derivatives.hessian_diagonal(module, g_inp, g_out)

        for sign in [self.PLUS, self.MINUS]:
            Hsign = clamp(sign * H, min=0, max=float('inf')).sqrt_()
            yield((diag_embed(Hsign), sign))

