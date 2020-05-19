from numpy import prod
from torch import clamp, diag_embed, einsum

from backpack.extensions.mat_to_mat_jac_base import MatToJacMat


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

        def diag_embed_multi_dim(H):
            """Convert [N, C_in, H_in, ...] to [N, C_in * H_in * ...,],
            embed into [N, C_in * H_in * ..., C_in * H_in = V], convert back
            to [V, N, C_in, H_in, ...,  V]."""
            feature_shapes = H.shape[1:]
            V, N = prod(feature_shapes), H.shape[0]

            H_diag = diag_embed(H.view(N, V))
            # [V, N, C_in, H_in, ...]
            shape = (V, N, *feature_shapes)
            return einsum("nic->cni", H_diag).view(shape)

        def decompose_into_positive_and_negative_sqrt(H):
            return [
                [diag_embed_multi_dim(positive_part(sign, H).sqrt_()), sign]
                for sign in [self.PLUS, self.MINUS]
            ]

        H = self.derivatives.hessian_diagonal(module, g_inp, g_out)
        return decompose_into_positive_and_negative_sqrt(H)
