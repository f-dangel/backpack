from torch import einsum

from backpack.core.derivatives.linear import LinearDerivatives
from backpack.extensions.secondorder.hbp.hbp_options import (
    BackpropStrategy,
    ExpectationApproximation,
)
from backpack.extensions.secondorder.hbp.hbpbase import HBPBaseModule


class HBPLinear(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=LinearDerivatives(), params=["weight", "bias"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            return self._weight_for_batch_average(ext, module, backproped)

        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._weight_for_sqrt(ext, module, backproped)

    def _weight_for_batch_average(self, ext, module, backproped):
        kron_factors = self._bias_for_batch_average(backproped)
        kron_factors += self._factors_from_input(ext, module)
        return kron_factors

    def _weight_for_sqrt(self, ext, module, backproped):
        kron_factors = self._factor_from_sqrt(backproped)
        kron_factors += self._factors_from_input(ext, module)
        return kron_factors

    def _factors_from_input(self, ext, module):
        ea_strategy = ext.get_ea_strategy()

        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            mean_input = self.__mean_input(module).unsqueeze(-1)
            return [mean_input, mean_input.transpose()]
        else:
            return [self.__mean_input_outer(module)]

    def _factor_from_sqrt(self, backproped):
        return [einsum("vni,vnj->ij", (backproped, backproped))]

    def bias(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            return self._bias_for_batch_average(backproped)
        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._factor_from_sqrt(backproped)

    def _bias_for_batch_average(self, backproped):
        return [backproped]

    def __mean_input(self, module):
        return module.input0.mean(0).flatten()

    def __mean_input_outer(self, module):
        N = module.input0.size(0)
        flat_input = module.input0.reshape(N, -1)
        return einsum("ni,nj->ij", (flat_input, flat_input)) / N
