import torch
from ...utils import conv as convUtils
from ....core.derivatives.conv2d import Conv2DDerivatives
from ....utils import einsum
from .cmpbase import CMPBase


class CMPConv2d(CMPBase, Conv2DDerivatives):
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

        def bias_cmp(mat):
            Jmat = self.bias_jac_mat_prod(module, grad_input, grad_output, mat)
            CJmat = CMP_out(Jmat)
            JTCJmat = self.bias_jac_t_mat_prod(module, grad_input, grad_output,
                                               CJmat)
            return JTCJmat

        return bias_cmp


EXTENSIONS = [CMPConv2d()]
