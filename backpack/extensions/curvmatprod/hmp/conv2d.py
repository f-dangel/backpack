from backpack.core.derivatives.conv2d import Conv2DDerivatives
from backpack.extensions.curvmatprod.hmp.hmpbase import HMPBase


class HMPConv2d(HMPBase):
    def __init__(self):
        super().__init__(derivatives=Conv2DDerivatives(), params=["weight", "bias"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        h_out_mat_prod = backproped

        def weight_hmp(mat):
            result = self.derivatives.weight_jac_mat_prod(module, g_inp, g_out, mat)
            result = h_out_mat_prod(result)
            result = self.derivatives.weight_jac_t_mat_prod(
                module, g_inp, g_out, result
            )

            return result

        return weight_hmp

    def bias(self, ext, module, g_inp, g_out, backproped):
        h_out_mat_prod = backproped

        def bias_hmp(mat):
            result = self.derivatives.bias_jac_mat_prod(module, g_inp, g_out, mat)
            result = h_out_mat_prod(result)
            result = self.derivatives.bias_jac_t_mat_prod(module, g_inp, g_out, result)

            return result

        return bias_hmp
