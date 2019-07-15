import torch
from math import sqrt
from .hbpbase import HBPBase
from ...utils.utils import einsum
from ...core.derivatives.linear import LinearDerivatives


class HBPLinear(HBPBase, LinearDerivatives):
    def __init__(self):
        super().__init__(params=["weight", "bias"])

    def weight(self, module, grad_input, grad_output):
        H_out = self.get_mat_from_ctx()

        kron_factors = []
        if self.AVG_PARAM_JAC is True:
            mean_input = self.__mean_input(module).unsqueeze(-1)
            kron_factors.append(mean_input)
            kron_factors.append(mean_input.transpose())
        else:
            kron_factors.append(self.__mean_input_outer(module))

        kron_factors.append(H_out)

        return kron_factors

    def bias(self, module, grad_input, grad_output):
        H_out = self.get_mat_from_ctx()

        kron_factors = [H_out]
        return kron_factors

    def __mean_input(self, module):
        _, flat_input = self.batch_flat(module.input0)
        return flat_input.mean(0)

    def __mean_input_outer(self, module):
        batch, flat_input = self.batch_flat(module.input0)
        # scale with sqrt
        flat_input_scaled /= sqrt(batch)
        return einsum('bi,bj->ij', (flat_input_scaled, flat_input_scaled))


EXTENSIONS = [HBPLinear()]
