from backpack.core.derivatives.conv2d import Conv2DDerivatives

from .cmpbase import CMPBase

from backpack.core.derivatives.utils import (
    weight_CMP_accept_vectors,
    bias_CMP_accept_vectors,
)


class CMPConv2d(CMPBase):
    def __init__(self):
        super().__init__(derivatives=Conv2DDerivatives(), params=["weight", "bias"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        CMP_out = backproped

        @weight_CMP_accept_vectors(module)
        def weight_cmp(mat):
            Jmat = self.derivatives.weight_jac_mat_prod(module, g_inp, g_out, mat)
            CJmat = CMP_out(Jmat)
            JTCJmat = self.derivatives.weight_jac_t_mat_prod(
                module, g_inp, g_out, CJmat
            )
            return JTCJmat

        return weight_cmp

    def bias(self, ext, module, g_inp, g_out, backproped):
        CMP_out = backproped

        @bias_CMP_accept_vectors(module)
        def bias_cmp(mat):
            Jmat = self.derivatives.bias_jac_mat_prod(module, g_inp, g_out, mat)
            CJmat = CMP_out(Jmat)
            JTCJmat = self.derivatives.bias_jac_t_mat_prod(module, g_inp, g_out, CJmat)
            return JTCJmat

        return bias_cmp
