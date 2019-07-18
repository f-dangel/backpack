from ...backpropextension import BackpropExtension
from ..strategies import BackpropStrategy
from ...context import get_from_ctx, set_in_ctx
from ...extensions import HBP
from ...curvature import Curvature


class HBPBase(BackpropExtension):
    MAT_NAME_IN_CTX = "_hbp_backpropagated_matrix"
    EXTENSION = HBP

    def __init__(self, params=None):
        if params is None:
            params = []
        super().__init__(self.get_module(), self.EXTENSION, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        M = self.get_mat_from_ctx()

        bp_strategy = self._get_bp_strategy_from_extension()
        if BackpropStrategy.is_batch_average(bp_strategy):
            M_mod = self.backpropagate_batch_average(module, grad_input,
                                                     grad_output, M)

        elif BackpropStrategy.is_sqrt(bp_strategy):
            M_mod = self.backpropagate_sqrt(module, grad_input, grad_output, M)

        self.set_mat_in_ctx(M_mod)

    def backpropagate_sqrt(self, module, grad_input, grad_output, H):
        return self.jac_t_mat_prod(module, grad_input, grad_output, H)

    def backpropagate_batch_average(self, module, grad_input, grad_output, H):
        ggn = self.ea_jac_t_mat_jac_prod(module, grad_input, grad_output, H)

        # second-order module effects
        residual = self._compute_residual_diag_if_nonzero(
            module, grad_input, grad_output)
        residual_mod = self._modify_residual(residual)

        if residual_mod is not None:
            ggn = self.add_diag_to_mat(residual_mod, ggn)

        return ggn

    def _compute_residual_diag_if_nonzero(self, module, grad_input,
                                          grad_output):
        if self.hessian_is_zero():
            return None

        if not self.hessian_is_diagonal():
            raise AttributeError(
                "Residual terms are only supported for elementwise functions")

        # second order module effects
        return self.hessian_diagonal(module, grad_input, grad_output).sum(0)

    def _modify_residual(self, residual):
        curv_type = self._get_curv_type_from_extension()
        return Curvature.modify_residual(residual, curv_type)

    def _get_curv_type_from_extension(self):
        return self._get_parametrized_ext().get_curv_type()

    def _get_bp_strategy_from_extension(self):
        return self._get_parametrized_ext().get_backprop_strategy()

    def _get_ea_strategy_from_extension(self):
        return self._get_parametrized_ext().get_ea_strategy()

    def get_mat_from_ctx(self):
        return get_from_ctx(self.MAT_NAME_IN_CTX)

    def set_mat_in_ctx(self, mat):
        set_in_ctx(self.MAT_NAME_IN_CTX, mat)

    @staticmethod
    def add_diag_to_mat(diag, mat):
        assert len(diag.shape) == 1
        assert len(mat.shape) == 2
        assert diag.shape[0] == mat.shape[0] == mat.shape[1]

        dim = diag.shape[0]
        idx = list(range(dim))

        mat[idx, idx] = mat[idx, idx] + diag
        return mat
