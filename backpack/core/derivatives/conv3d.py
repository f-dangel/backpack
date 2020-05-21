from torch.nn import Conv3d

from backpack.core.derivatives.basederivatives import BaseParameterDerivatives


class Conv3DDerivatives(BaseParameterDerivatives):
    def get_module(self):
        return Conv3d

    def hessian_is_zero(self):
        return True

    # TODO: @sbharadwajj implement and test
    def get_unfolded_input(self, module):
        raise NotImplementedError

    # TODO: @sbharadwajj implement and test
    def ea_jac_t_mat_jac_prod(self, module, g_inp, g_out, mat):
        raise NotImplementedError

    # TODO: @sbharadwajj implement and test
    def _jac_mat_prod(self, module, g_inp, g_out, mat):
        raise NotImplementedError

    # TODO: @sbharadwajj implement and test
    def _jac_t_mat_prod(self, module, g_inp, g_out, mat):
        raise NotImplementedError

    # TODO: @sbharadwajj implement and test
    def _bias_jac_mat_prod(self, module, g_inp, g_out, mat):
        raise NotImplementedError

    # TODO: @sbharadwajj implement and test
    def _bias_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        raise NotImplementedError

    # TODO: Improve performance by using conv instead of unfold

    # TODO: @sbharadwajj implement and test
    def _weight_jac_mat_prod(self, module, g_inp, g_out, mat):
        raise NotImplementedError

    # TODO: @sbharadwajj implement and test
    def _weight_jac_t_mat_prod(self, module, g_inp, g_out, mat, sum_batch=True):
        raise NotImplementedError
