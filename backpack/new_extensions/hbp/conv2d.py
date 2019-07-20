from ...utils import conv as convUtils
from ...core.derivatives.conv2d import Conv2DDerivatives, Conv2DConcatDerivatives
from ...utils.utils import einsum
from .hbpbase import HBPBaseModule
from .hbp_options import BackpropStrategy, ExpectationApproximation


class HBPConv2d(HBPBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DDerivatives(),
            params=["weight", "bias"]
        )

    def weight(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            raise NotImplementedError

        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._weight_for_sqrt(ext, module, g_inp, g_out, backproped)

    def _weight_for_sqrt(self, ext, module, g_inp, g_out, backproped):
        kron_factors = [
            self._factor_from_sqrt(ext, module, g_inp, g_out, backproped)
        ]
        kron_factors += self._factors_from_input(
            ext, module, g_inp, g_out, backproped
        )
        return kron_factors

    def _factors_from_input(self, ext, module, g_inp, g_out, backproped):
        X = convUtils.unfold_func(module)(module.input0)
        batch = X.size(0)

        ea_strategy = ext.get_ea_strategy()

        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            raise NotImplementedError
        else:
            yield einsum('bik,bjk->ij', (X, X)) / batch

    def _factor_from_sqrt(self, ext, module, g_inp, g_out, backproped):
        sqrt_ggn = backproped
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn)
        sqrt_ggn = einsum('bijc->bic', (sqrt_ggn,))
        return einsum('bic,blc->il', (sqrt_ggn, sqrt_ggn))

    def bias(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            raise NotImplementedError

        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._bias_for_sqrt(ext, module, g_inp, g_out, backproped)

    def _bias_for_sqrt(self, ext, module, g_inp, g_out, backproped):
        return [self._factor_from_sqrt(ext, module, g_inp, g_out, backproped)]


class HBPConv2dConcat(HBPBaseModule):
    def __init__(self):
        super().__init__(
            derivatives=Conv2DConcatDerivatives(),
            params=["weight"]
        )

    def weight(self, ext, module, g_inp, g_out, backproped):
        bp_strategy = ext.get_backprop_strategy()

        if BackpropStrategy.is_batch_average(bp_strategy):
            raise NotImplementedError
        elif BackpropStrategy.is_sqrt(bp_strategy):
            return self._weight_for_sqrt(ext, module, g_inp, g_out, backproped)

    def _weight_for_sqrt(self, ext, module, g_inp, g_out, backproped):
        kron_factors = [
            self._factor_from_sqrt(ext, module, g_inp, g_out, backproped)
        ]
        kron_factors += self._factors_from_input(
            ext, module, g_inp, g_out, backproped
        )

        return kron_factors

    def _factors_from_input(self, ext, module, g_inp, g_out, backproped):
        X = module.homogeneous_unfolded_input()
        batch = X.size(0)

        ea_strategy = ext.get_ea_strategy()

        if ExpectationApproximation.should_average_param_jac(ea_strategy):
            raise NotImplementedError
        else:
            yield einsum('bik,bjk->ij', (X, X)) / batch

    def _factor_from_sqrt(self, ext, module, g_inp, g_out, backproped):
        sqrt_ggn = backproped
        sqrt_ggn = convUtils.separate_channels_and_pixels(module, sqrt_ggn)
        sqrt_ggn = einsum('bijc->bic', (sqrt_ggn,))
        return einsum('bic,blc->il', (sqrt_ggn, sqrt_ggn))


EXTENSIONS = [HBPConv2d(), HBPConv2dConcat()]
