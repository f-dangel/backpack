from backpack.core.derivatives.utils import hmp_unsqueeze_if_missing_dim
from backpack.extensions.curvature import Curvature
from backpack.extensions.module_extension import ModuleExtension
from backpack.utils.utils import einsum


class CMPBase(ModuleExtension):
    """
    Given matrix-vector product routine `MVP(A)` backpropagate to `MVP(J^T A J)`.
    """
    def __init__(self, derivatives, params=None):
        super().__init__(params=params)
        self.derivatives = derivatives

    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        CMP_out = backproped

        residual = self._second_order_module_effects(module, ext, g_inp, g_out)
        residual_mod = self._modify_residual(ext, residual)

        @hmp_unsqueeze_if_missing_dim(mat_dim=3)
        def CMP_in(mat):
            """Multiplication of curvature matrix with matrix `mat`.

            Parameters:
            -----------
            mat : torch.Tensor
                Matrix that will be multiplied.
            """
            Jmat = self.derivatives.jac_mat_prod(module, g_inp, g_out, mat)
            CJmat = CMP_out(Jmat)
            JTCJmat = self.derivatives.jac_t_mat_prod(module, g_inp, g_out,
                                                      CJmat)

            if residual_mod is not None:
                JTCJmat.add_(einsum('bi,bic->bic', (residual_mod, mat)))

            return JTCJmat

        return CMP_in

    def _second_order_module_effects(self, module, ext, g_inp, g_out):
        if self.derivatives.hessian_is_zero():
            return None
        if not Curvature.require_residual(ext.get_curv_type()):
            return None

        if not self.derivatives.hessian_is_diagonal():
            raise NotImplementedError(
                "Residual terms are only supported for elementwise functions")

        return self.derivatives.hessian_diagonal(module, g_inp, g_out)

    def _modify_residual(self, ext, residual):
        return Curvature.modify_residual(residual, ext.get_curv_type())
