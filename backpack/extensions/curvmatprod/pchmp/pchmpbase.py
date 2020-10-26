"""Block positive curvature Hessian-matrix products"""

import torch

from backpack.extensions.module_extension import ModuleExtension


class PCHMPBase(ModuleExtension):
    def __init__(self, derivatives, params=None):
        super().__init__(params=params)
        self.derivatives = derivatives

        self.modifications = {
            "abs": self.to_abs,
            "clip": self.zero_negative_elements,
        }

    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        """Backpropagate Hessian multiplication routines.

        Given mat → ℋz(x) mat, backpropagate mat → ℋx mat.
        """
        diagonal_or_zero_residual = (
            self.derivatives.hessian_is_zero() or self.derivatives.hessian_is_diagonal()
        )
        if not diagonal_or_zero_residual:
            raise ValueError("Only linear or element-wise operations supported.")

        h_out_mat_prod = backproped

        modify = ext.get_modification_mode()

        def h_in_mat_prod(mat):
            """Multiplication with curvature matrix w.r.t. the module input.

            Parameters:
            -----------
            mat : torch.Tensor
                Matrix that will be multiplied.
            modify (str): how to modify the residual terms
            """
            # Multiply with the GGN term: mat → [𝒟z(x)ᵀ ℋz 𝒟z(x)] mat.
            result = self.derivatives.jac_mat_prod(module, g_inp, g_out, mat)
            result = h_out_mat_prod(result)
            result = self.derivatives.jac_t_mat_prod(module, g_inp, g_out, result)

            # Multiply with the residual term: mat → [∑ᵢ Hzᵢ(x) δzᵢ] mat.
            if not self.derivatives.hessian_is_zero():
                result += self.modified_residual_mat_prod(
                    ext, module, g_inp, g_out, mat, modify
                )
            return result

        return h_in_mat_prod

    # TODO: Add shape check and accept vectors
    def modified_residual_mat_prod(self, ext, module, g_inp, g_out, mat, modify):
        if modify not in self.modifications.keys():
            raise KeyError(
                "Supported modes: {}, but got {}".format(
                    self.modifications.keys(), modify
                )
            )

        residual = self.derivatives.hessian_diagonal(module, g_inp, g_out) * g_out[0]
        residual = self.modifications[modify](residual)

        return torch.einsum("...,v...->v...", (residual, mat))

    @staticmethod
    def zero_negative_elements(tensor):
        return tensor.clamp(min=0)

    @staticmethod
    def to_abs(tensor):
        return tensor.abs()
