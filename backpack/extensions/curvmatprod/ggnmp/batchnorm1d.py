from backpack.core.derivatives.batchnorm1d import BatchNorm1dDerivatives
from backpack.extensions.curvmatprod.ggnmp.ggnmpbase import GGNMPBase


class GGNMPBatchNorm1d(GGNMPBase):
    def __init__(self):
        super().__init__(
            derivatives=BatchNorm1dDerivatives(), params=["weight", "bias"]
        )

    def weight(self, ext, module, g_inp, g_out, backproped):
        h_out_mat_prod = backproped

        def weight_ggnmp(mat):
            result = self.derivatives.weight_jac_mat_prod(module, g_inp, g_out, mat)
            result = h_out_mat_prod(result)
            result = self.derivatives.weight_jac_t_mat_prod(
                module, g_inp, g_out, result
            )

            return result

        return weight_ggnmp

    def bias(self, ext, module, g_inp, g_out, backproped):
        h_out_mat_prod = backproped

        def bias_ggnmp(mat):
            result = self.derivatives.bias_jac_mat_prod(module, g_inp, g_out, mat)
            result = h_out_mat_prod(result)
            result = self.derivatives.bias_jac_t_mat_prod(module, g_inp, g_out, result)

            return result

        return bias_ggnmp
