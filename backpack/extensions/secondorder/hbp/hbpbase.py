from backpack.extensions.curvature import Curvature
from backpack.extensions.module_extension import ModuleExtension
from backpack.extensions.secondorder.hbp.hbp_options import BackpropStrategy


class HBPBaseModule(ModuleExtension):
    def __init__(self, derivatives, params=None):
        self.derivatives = derivatives
        super().__init__(params=params)

    def backpropagate(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            return self.backpropagate_batch_average(
                ext, module, g_inp, g_out, backproped
            )

        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self.backpropagate_sqrt(ext, module, g_inp, g_out, backproped)

    def backpropagate_sqrt(self, ext, module, g_inp, g_out, H):
        return self.derivatives.jac_t_mat_prod(module, g_inp, g_out, H)

    def backpropagate_batch_average(self, ext, module, g_inp, g_out, H):
        ggn = self.derivatives.ea_jac_t_mat_jac_prod(module, g_inp, g_out, H)

        residual = self.second_order_module_effects(module, g_inp, g_out)
        residual_mod = Curvature.modify_residual(residual, ext.get_curv_type())

        if residual_mod is not None:
            ggn = self.add_diag_to_mat(residual_mod, ggn)

        return ggn

    def second_order_module_effects(self, module, g_inp, g_out):
        if self.derivatives.hessian_is_zero():
            return None

        elif not self.derivatives.hessian_is_diagonal():
            raise NotImplementedError(
                "Residual terms are only supported for elementwise functions"
            )

        else:
            return self.derivatives.hessian_diagonal(module, g_inp, g_out).sum(0)

    @staticmethod
    def add_diag_to_mat(diag, mat):
        assert len(diag.shape) == 1
        assert len(mat.shape) == 2
        assert diag.shape[0] == mat.shape[0] == mat.shape[1]

        dim = diag.shape[0]
        idx = list(range(dim))

        mat[idx, idx] = mat[idx, idx] + diag
        return mat
