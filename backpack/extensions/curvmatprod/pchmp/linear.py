from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.curvmatprod.pchmp.pchmpbase import PCHMPBase


class PCHMPLinear(PCHMPBase):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["weight", "bias"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        h_out_mat_prod = backproped

        def weight_pchmp(mat):
            result = self.derivatives.weight_jac_mat_prod(module, g_inp, g_out, mat)
            result = h_out_mat_prod(result)
            result = self.derivatives.param_mjp("weight", module, g_inp, g_out, result)

            return result

        return weight_pchmp

    def bias(self, ext, module, g_inp, g_out, backproped):
        h_out_mat_prod = backproped

        def bias_pchmp(mat):
            result = self.derivatives.bias_jac_mat_prod(module, g_inp, g_out, mat)
            result = h_out_mat_prod(result)
            result = self.derivatives.param_mjp("bias", module, g_inp, g_out, result)

            return result

        return bias_pchmp
