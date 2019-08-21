import warnings

from backpack.core.derivatives.conv2d import (Conv2DConcatDerivatives,
                                              Conv2DDerivatives)
from backpack.utils import conv as convUtils
from backpack.utils.utils import einsum

from ....utils.utils import random_psd_matrix
from .hbp_options import BackpropStrategy, ExpectationApproximation
from .hbpbase import HBPBaseModule


class HBPConv2d(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=Conv2DDerivatives(),
                         params=["weight", "bias"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            return self._weight_for_batch_average(ext, module, backproped)

        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._weight_for_sqrt(ext, module, backproped)

    def _weight_for_batch_average(self, ext, module, backproped):
        """CAUTION: Return a random PSD matrix."""
        warnings.warn("[DUMMY IMPLEMENTATION] KFRA weight for Conv2d")
        out_c, in_c, k_x, k_y = module.weight.size()
        device = module.weight.device
        return [
            random_psd_matrix(out_c, device=device),
            random_psd_matrix(in_c * k_x * k_y, device=device)
        ]

    def _weight_for_sqrt(self, ext, module, backproped):
        kron_factors = [self._factor_from_sqrt(module, backproped)]
        kron_factors += self._factors_from_input(ext, module)
        return kron_factors

    def _factors_from_input(self, ext, module):
        X = convUtils.unfold_func(module)(module.input0)
        batch = X.size(0)

        ea_strategy = ext.get_ea_strategy()

        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            raise NotImplementedError
        else:
            yield einsum('bik,bjk->ij', (X, X)) / batch

    def _factor_from_sqrt(self, module, backproped):
        sqrt_ggn = backproped
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn)
        sqrt_ggn = einsum('bijc->bic', (sqrt_ggn, ))
        return einsum('bic,blc->il', (sqrt_ggn, sqrt_ggn))

    def bias(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            return self._bias_for_batch_average(module, backproped)
        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._bias_for_sqrt(module, backproped)

    def _bias_for_sqrt(self, module, backproped):
        return [self._factor_from_sqrt(module, backproped)]

    def _bias_for_batch_average(self, module, backproped):
        """CAUTION: Return a random PSD matrix."""
        warnings.warn("[DUMMY IMPLEMENTATION] KFRA bias for Conv2d")
        bias_dim = module.bias.numel()
        device = module.bias.device
        return [random_psd_matrix(bias_dim, device=device)]


class HBPConv2dConcat(HBPBaseModule):
    def __init__(self):
        super().__init__(derivatives=Conv2DConcatDerivatives(),
                         params=["weight"])

    def weight(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            raise NotImplementedError
        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._weight_for_sqrt(ext, module, backproped)

    def _weight_for_sqrt(self, ext, module, backproped):
        kron_factors = [self._factor_from_sqrt(module, backproped)]
        kron_factors += self._factors_from_input(ext, module)

        return kron_factors

    def _factors_from_input(self, ext, module):
        X = module.homogeneous_unfolded_input()
        batch = X.size(0)

        ea_strategy = ext.get_ea_strategy()

        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            raise NotImplementedError
        else:
            yield einsum('bik,bjk->ij', (X, X)) / batch

    def _factor_from_sqrt(self, module, backproped):
        sqrt_ggn = backproped
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn)
        sqrt_ggn = einsum('bijc->bic', (sqrt_ggn, ))
        return einsum('bic,blc->il', (sqrt_ggn, sqrt_ggn))


EXTENSIONS = [HBPConv2d(), HBPConv2dConcat()]
