import torch
from ...utils import conv as convUtils
from ...core.derivatives.conv2d import Conv2DDerivatives
from ...utils.utils import einsum
from .hbpbase import HBPBase
from ..strategies import BackpropStrategy, LossHessianStrategy, ExpectationApproximation


class HBPConv2d(HBPBase, Conv2DDerivatives):
    def __init__(self):
        super().__init__(params=["weight", "bias"])

    # WEIGHT
    ###
    def weight(self, module, grad_input, grad_output):
        if BackpropStrategy.is_batch_average():
            raise NotImplementedError
        elif BackpropStrategy.is_sqrt():
            return self._weight_for_sqrt(module, grad_input, grad_output)

    def _weight_for_sqrt(self, module, grad_input, grad_output):
        kron_factors = []

        if LossHessianStrategy.is_kfac():
            kron_factors.append(
                self._factor_from_sqrt_sampling(module, grad_input,
                                                grad_output))

        elif LossHessianStrategy.is_kflr():
            kron_factors.append(
                self._factor_from_sqrt_exact(module, grad_input, grad_output))

        for factor in self._factors_from_input(module, grad_input,
                                               grad_output):
            kron_factors.append(factor)

        return kron_factors

    def _factors_from_input(self, module, grad_input, grad_output):
        X = convUtils.unfold_func(module)(module.input0)
        batch = X.size(0)

        if ExpectationApproximation.should_average_param_jac():
            raise NotImplementedError("This EA is not defined by Chen")
        else:
            yield einsum('bik,bjk->ij', (X, X)) / batch

    def _factor_from_sqrt_exact(self, module, grad_input, grad_output):
        sqrt_ggn = self.get_mat_from_ctx()
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn)
        sqrt_ggn = einsum('bijc->bic', (sqrt_ggn, ))
        return einsum('bic,blc->il', (sqrt_ggn, sqrt_ggn))

    def _factor_from_sqrt_sampling(self, module, grad_input, grad_output):
        sqrt_mc = self.get_mat_from_ctx()
        samples = sqrt_mc.size(2)
        sqrt_mc = convUtils.separate_channels_and_pixels(module, sqrt_mc)
        sqrt_mc = einsum('bijc->bic', (sqrt_mc, ))
        # TODO: Divide by samples correct?
        return einsum('bic,blc->il', (sqrt_mc, sqrt_mc)) / samples

    ###

    # BIAS
    ###
    def bias(self, module, grad_input, grad_output):
        if BackpropStrategy.is_batch_average():
            raise NotImplementedError
        elif BackpropStrategy.is_sqrt():
            return self._bias_for_sqrt(module, grad_input, grad_output)

    def _bias_for_sqrt(self, module, grad_input, grad_output):
        kron_factors = []

        if LossHessianStrategy.is_kfac():
            kron_factors.append(
                self._factor_from_sqrt_sampling(module, grad_input,
                                                grad_output))

        elif LossHessianStrategy.is_kflr():
            kron_factors.append(
                self._factor_from_sqrt_exact(module, grad_input, grad_output))

        return kron_factors


EXTENSIONS = [HBPConv2d()]
