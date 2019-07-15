from ...extensions import HBP
from ...matbackprop import ExpectationApproximationMatToJacMatJac
from ...curvature import Curvature
from ...context import set_in_ctx, get_from_ctx


class HBPBase(ExpectationApproximationMatToJacMatJac):
    MAT_NAME_IN_CTX = "_hbp_backpropagated_hessian_ea"
    EXTENSION = HBP

    def __init__(self, params=None):
        """Backprop batch average (expectation approximation) of Hessian."""
        if params is None:
            params = []
        super().__init__(self.MAT_NAME_IN_CTX, self.EXTENSION, params=params)

    def backpropagate(self, module, grad_input, grad_output):
        ea_H_out = self.get_mat_from_ctx()

        ea_H_in = self.backpropagate_ggn_term(module, grad_input, grad_output,
                                              ea_H_out)
        ea_H_in = self.add_residual_term(module, grad_input, grad_output,
                                         ea_H_in)
        self.set_mat_in_ctx(ea_H_in)

    def backpropagate_ggn_term(self, module, grad_input, grad_output,
                               ea_h_out):
        """Given EA of the output Hessian, compute EA of the input Hessian."""
        return self.ea_jac_t_mat_jac_prod(module, grad_input, grad_output,
                                          ea_h_out)

    def add_residual_term(self, module, grad_input, grad_output, mat):
        """Second-order effects introduced by the module function."""
        ea_residual = self._compute_ea_residual_diag_if_nonzero(
            module, grad_input, grad_output)
        ea_residual_mod = Curvature.modify_residual(ea_residual)
        if ea_residual_mod is not None:
            mat = self.add_on_diag(mat, ea_residual_mod)
        return mat

    def _compute_ea_residual_diag_if_nonzero(self, module, grad_input,
                                             grad_output):
        if self.hessian_is_zero():
            return None

        if not self.hessian_is_diagonal():
            raise AttributeError(
                "Residual terms are only supported for elementwise functions")

        return self.hessian_diagonal(module, grad_input, grad_output).sum(0)

    @staticmethod
    def add_on_diag(mat, diag):
        dim = mat.size(0)
        idx = list(range(dim))
        mat[idx, idx] = mat[idx, idx] + diag
        return mat

    def get_mat_from_ctx(self):
        return get_from_ctx(self.MAT_NAME_IN_CTX)

    def set_mat_in_ctx(self, mat):
        print('HI')
        print(self.MAT_NAME_IN_CTX)
        print('HI')
        set_in_ctx(self.MAT_NAME_IN_CTX, mat)
