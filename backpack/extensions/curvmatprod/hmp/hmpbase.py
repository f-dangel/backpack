"""Block Hessian-matrix products"""

from backpack.extensions.module_extension import ModuleExtension


class HMPBase(ModuleExtension):
    def __init__(self, derivatives, params=None):
        super().__init__(params=params)
        self.derivatives = derivatives

    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        """Backpropagate Hessian multiplication routines.

        Given mat → ℋz(x) mat, backpropagate mat → ℋx mat.
        """
        h_out_mat_prod = backproped

        def h_in_mat_prod(mat):
            """Multiplication with curvature matrix w.r.t. the module input.

            Parameters:
            -----------
            mat : torch.Tensor
                Matrix that will be multiplied.
            """
            # Multiply with the GGN term: mat → [𝒟z(x)ᵀ ℋz 𝒟z(x)] mat.
            result = self.derivatives.jac_mat_prod(module, g_inp, g_out, mat)
            result = h_out_mat_prod(result)
            result = self.derivatives.jac_t_mat_prod(module, g_inp, g_out, result)

            # Multiply with the residual term: mat → [∑ᵢ Hzᵢ(x) δzᵢ] mat.
            if not self.derivatives.hessian_is_zero():
                result += self.derivatives.residual_mat_prod(module, g_inp, g_out, mat)

            return result

        return h_in_mat_prod
