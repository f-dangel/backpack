import torch
from math import sqrt
from .hbpbase import HBPBase
from ...utils.utils import einsum
from ...core.derivatives.linear import LinearDerivatives, LinearConcatDerivatives
from ..strategies import BackpropStrategy, LossHessianStrategy, ExpectationApproximation


class HBPLinear(HBPBase, LinearDerivatives):
    def __init__(self):
        super().__init__(params=["weight", "bias"])

    # WEIGHT
    ###
    def weight(self, module, grad_input, grad_output):
        bp_strategy = self._get_bp_strategy_from_extension()
        if BackpropStrategy.is_batch_average(bp_strategy):
            return self._weight_for_batch_average(module, grad_input,
                                                  grad_output)
        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._weight_for_sqrt(module, grad_input, grad_output)

    def _weight_for_batch_average(self, module, grad_input, grad_output):
        kron_factors = self._bias_for_batch_average(module, grad_input,
                                                    grad_output)

        for factor in self._factors_from_input(module, grad_input,
                                               grad_output):
            kron_factors.append(factor)

        return kron_factors

    def _weight_for_sqrt(self, module, grad_input, grad_output):
        kron_factors = [
            self._factor_from_sqrt(module, grad_input, grad_output)
        ]

        for factor in self._factors_from_input(module, grad_input,
                                               grad_output):
            kron_factors.append(factor)

        return kron_factors

    def _factors_from_input(self, module, grad_input, grad_output):
        ea_strategy = self._get_ea_strategy_from_extension()
        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            mean_input = self.__mean_input(module).unsqueeze(-1)
            yield mean_input
            yield mean_input.transpose()
        else:
            yield self.__mean_input_outer(module)

    def _factor_from_sqrt(self, module, grad_input, grad_output):
        sqrt_ggn_out = self.get_mat_from_ctx()
        return einsum('bic,bjc->ij', (sqrt_ggn_out, sqrt_ggn_out))

    ###

    # BIAS
    ###
    def bias(self, module, grad_input, grad_output):
        bp_strategy = self._get_bp_strategy_from_extension()
        if BackpropStrategy.is_batch_average(bp_strategy):
            return self._bias_for_batch_average(module, grad_input,
                                                grad_output)
        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._bias_for_sqrt(module, grad_input, grad_output)

    def _bias_for_batch_average(self, module, grad_input, grad_output):
        kron_factors = [self.get_mat_from_ctx()]
        return kron_factors

    def _bias_for_sqrt(self, module, grad_input, grad_output):
        return [self._factor_from_sqrt(module, grad_input, grad_output)]

    def __mean_input(self, module):
        _, flat_input = self.batch_flat(module.input0)
        return flat_input.mean(0)

    def __mean_input_outer(self, module):
        batch, flat_input = self.batch_flat(module.input0)
        # scale with sqrt
        flat_input /= sqrt(batch)
        return einsum('bi,bj->ij', (flat_input, flat_input))


class HBPLinearConcat(HBPBase, LinearConcatDerivatives):
    def __init__(self):
        super().__init__(params=["weight"])

    # WEIGHT
    ###
    def weight(self, module, grad_input, grad_output):
        bp_strategy = self._get_bp_strategy_from_extension()
        if BackpropStrategy.is_batch_average(bp_strategy):
            return self._weight_for_batch_average(module, grad_input,
                                                  grad_output)
        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._weight_for_sqrt(module, grad_input, grad_output)

    def _weight_for_batch_average(self, module, grad_input, grad_output):
        kron_factors = self._bias_for_batch_average(module, grad_input,
                                                    grad_output)

        for factor in self._factors_from_input(module, grad_input,
                                               grad_output):
            kron_factors.append(factor)

        return kron_factors

    def _weight_for_sqrt(self, module, grad_input, grad_output):
        kron_factors = [
            self._factor_from_sqrt(module, grad_input, grad_output)
        ]

        for factor in self._factors_from_input(module, grad_input,
                                               grad_output):
            kron_factors.append(factor)

        return kron_factors

    def _factors_from_input(self, module, grad_input, grad_output):
        ea_strategy = self._get_ea_strategy_from_extension()
        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            mean_input = self.__mean_input(module).unsqueeze(-1)
            yield mean_input
            yield mean_input.transpose()
        else:
            yield self.__mean_input_outer(module)

    def _factor_from_sqrt(self, module, grad_input, grad_output):
        sqrt_ggn_out = self.get_mat_from_ctx()
        return einsum('bic,bjc->ij', (sqrt_ggn_out, sqrt_ggn_out))

    ###

    def __mean_input(self, module):
        _, flat_input = self.batch_flat(module.homogeneous_input())
        return flat_input.mean(0)

    def __mean_input_outer(self, module):
        batch, flat_input = self.batch_flat(module.homogeneous_input())
        # scale with sqrt
        flat_input /= sqrt(batch)
        return einsum('bi,bj->ij', (flat_input, flat_input))


EXTENSIONS = [HBPLinear(), HBPLinearConcat()]
