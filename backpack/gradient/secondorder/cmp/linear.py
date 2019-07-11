import torch
from .cmpbase import CMPBase
from ....utils import einsum
from ....core.derivatives.linear import LinearDerivatives


class CMPLinear(CMPBase, LinearDerivatives):
    def __init__(self):
        super().__init__(params=["weight", "bias"])

    def weight(self, module, grad_input, grad_output):
        CMP_out = self.get_from_ctx()

        def weight_cmp(mat):
            Jmat = self.weight_jac_mat_prod(module, grad_input, grad_output,
                                            mat)
            CJmat = CMP_out(Jmat)
            JTCJmat = self.weight_jac_t_mat_prod(module, grad_input,
                                                 grad_output, CJmat)
            return JTCJmat

        return weight_cmp

    def bias(self, module, grad_input, grad_output):
        CMP_out = self.get_from_ctx()

        def bias_cmp(mat, which=None):
            Jmat = self.bias_jac_mat_prod(module, grad_input, grad_output, mat)
            CJmat = CMP_out(Jmat)
            JTCJmat = self.bias_jac_t_mat_prod(module, grad_input, grad_output,
                                               CJmat)
            return JTCJmat

        return bias_cmp


EXTENSIONS = [CMPLinear()]
