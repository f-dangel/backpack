import torch
from .cmpbase import CMPBase
from ....utils import einsum
from ...derivatives.linear import LinearDerivatives


class CMPLinear(CMPBase, LinearDerivatives):
    def __init__(self):
        super().__init__(params=["weight", "bias"])

    def weight(self, module, grad_input, grad_output):
        CMP_out = self.get_from_ctx()

        print('\nCreate weight HMP')

        def weight_hmp(mat):
            Jmat = self.weight_jac_mat_prod(module, grad_input, grad_output,
                                            mat)
            print('Weight calling CMP with id', id(CMP_out))
            CJmat = CMP_out(Jmat)
            JTCJmat = self.weight_jac_t_mat_prod(module, grad_input,
                                                 grad_output, CJmat)
            return JTCJmat

        print('Finish creation of weight HMP\n')

        return weight_hmp

    def bias(self, module, grad_input, grad_output):
        CMP_out = self.get_from_ctx()

        print('\nCreate bias HMP')

        def bias_hmp(mat):
            Jmat = self.bias_jac_mat_prod(module, grad_input, grad_output, mat)
            print('Bias calling CMP with id', id(CMP_out))
            CJmat = CMP_out(Jmat)
            JTCJmat = self.bias_jac_t_mat_prod(module, grad_input, grad_output,
                                               CJmat)
            return JTCJmat

        print('Finish creation of bias HMP\n')

        return bias_hmp


EXTENSIONS = [CMPLinear()]
