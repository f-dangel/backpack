from torch.nn import Flatten


from backpack.core.derivatives.basederivatives import BaseDerivatives


class FlattenDerivatives(BaseDerivatives):
    def get_module(self):
        return Flatten

    def hessian_is_zero(self):
        return True

    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        return mat

    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        return self.view_like_input(mat, module)

    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        return self.view_like_output(mat, module)
