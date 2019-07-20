from backpack.new_extensions.newmatbackprop import MatToJacMat
from torch import clamp, diag_embed


class DiagHBaseModule(MatToJacMat):
    PLUS = 1.
    MINUS = -1.

    def __init__(self, derivatives, params=None):
        super().__init__(derivatives=derivatives, params=params)

    def backpropagate(self, ext, module, grad_input, grad_output, backproped):
        bp_matrices = backproped["matrices"]
        bp_signs = backproped["signs"]

        bp_matrices = super().backpropagate(ext, module, grad_input, grad_output, bp_matrices)

        for matrix, sign in self.local_curvatures(module, grad_input, grad_output):
            bp_matrices.append(matrix)
            bp_signs.append(sign)

        return {
            "matrices": bp_matrices,
            "signs": bp_signs
        }

    def local_curvatures(self, module, grad_input, grad_output):
        if self.derivatives is None or self.derivatives.hessian_is_zero():
            return []
        if not self.derivatives.hessian_is_diagonal():
            raise NotImplementedError

        H = self.derivatives.hessian_diagonal(module, grad_input, grad_output)

        for sign in [self.PLUS, self.MINUS]:
            Hsign = clamp(sign * H, min=0, max=float('inf')).sqrt_()
            yield((diag_embed(Hsign), sign))

