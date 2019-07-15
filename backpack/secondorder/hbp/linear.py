import torch
from math import sqrt
from .hbpbase import HBPBase
from ...utils.utils import einsum
from ...core.derivatives.linear import LinearDerivatives
from ..ea import ExpectationApproximation


class HBPLinear(HBPBase, LinearDerivatives):
    def __init__(self):
        super().__init__(params=["weight", "bias"])

    def weight(self, module, grad_input, grad_output):
        H_out = self.get_mat_from_ctx()

        kron_factors = [H_out]
        if ExpectationApproximation.should_average_param_jac():
            mean_input = self.__mean_input(module).unsqueeze(-1)
            kron_factors.append(mean_input)
            kron_factors.append(mean_input.transpose())
        else:
            kron_factors.append(self.__mean_input_outer(module))

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
        flat_input /= sqrt(batch)
        return einsum('bi,bj->ij', (flat_input, flat_input))


EXTENSIONS = [HBPLinear()]
