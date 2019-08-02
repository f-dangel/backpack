from backpack.core.derivatives.linear import (LinearConcatDerivatives,
                                              LinearDerivatives)

from .cmpbase import CMPBase


class CMPLinear(CMPBase):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(),
                         params=["weight", "bias"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        CMP_out = backproped

        def weight_cmp(mat):
            Jmat = self.derivatives.weight_jac_mat_prod(
                module, g_inp, g_out, mat)
            CJmat = CMP_out(Jmat)
            JTCJmat = self.derivatives.weight_jac_t_mat_prod(
                module, g_inp, g_out, CJmat)
            return JTCJmat

        return weight_cmp

    def bias(self, ext, module, g_inp, g_out, backproped):
        CMP_out = backproped

        def bias_cmp(mat):
            Jmat = self.derivatives.bias_jac_mat_prod(module, g_inp, g_out,
                                                      mat)
            CJmat = CMP_out(Jmat)
            JTCJmat = self.derivatives.bias_jac_t_mat_prod(
                module, g_inp, g_out, CJmat)
            return JTCJmat

        return bias_cmp


class CMPLinearConcat(CMPBase):
    def __init__(self):
        super().__init__(derivatives=LinearConcatDerivatives(),
                         params=["weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        CMP_out = backproped

        def weight_cmp(mat):
            Jmat = self.derivatives.weight_jac_mat_prod(
                module, g_inp, g_out, mat)
            CJmat = CMP_out(Jmat)
            JTCJmat = self.derivatives.weight_jac_t_mat_prod(
                module, g_inp, g_out, CJmat)
            return JTCJmat

        return weight_cmp
